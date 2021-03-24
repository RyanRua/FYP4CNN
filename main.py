import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from datetime import datetime
import pdb
import random
import sys, os, time
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.duplicate import ElementwiseDuplicateElimination
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.algorithms.moead import MOEAD
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.algorithms.ctaea import CTAEA
import copy
from sklearn.cluster import KMeans
import csv
import os
import argparse
import torch
from torch import nn

# Default model parameters
# Number of cluster in kmeans
num_cluster = 20
# Asset name
symbols = []
# Concrete dataset
data = []
# Clustering result
clusters = []
# Time interval for objective calculation
interval = 7
# Risk threshold(percentage)
rThreshold = 100
# Risk factor
rFactor = 0.9


# MOEA parameters
# Probability of crossover
crossover_rate = 0.7
# Probability of mutation
mutaion_rate = 0.3
# Population size
p_size = 200
# Number of generation
n_gen = 50
# Partition number of reference direction
n_partitions = 50 
# MOEA/D decomposition approach. Support 'pbi' and 'tchebi'
decomposition = 'tchebi'
# Number of neighbors in MOEA/D
n_neighbors = 20
# MOEA algorithm
algorithm_name = 'NSGA3'

# Device for NN
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Optimization parameters
range_rFactor = (0.5, 0.95)
range_cluster = (5, 50)
range_partitions = (50,300)
epoch = 100


# Gloabl variable
# Temp objectives 
tempObjectives = []
# Temp chromosomes
tempChromosomes = []


def eval(portfolio,weights):

    # Sum for each time interval
    df = {}
    for asset in portfolio:
        temp = np.array(data[asset])
        temp_slice = temp[0:len(temp):interval].tolist();
        df[asset] = temp_slice
    df = pd.DataFrame(df)
    my_stocks = df
    #Show the daily simple returns, Formula = new_price/old_price - 1
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 252
    #Expected portfolio variance= WT * (Covariance Matrix) * W
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    port_standardD = np.sqrt(port_variance)
    portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights) * 252

    percent_var = round(port_variance, 8) * 100
    percent_std = round(port_standardD, 8) * 100
    percent_ret = round(portfolioSimpleAnnualReturn, 8)*100

    return [-percent_ret, percent_std]
    


# Random initialization with random asset in each clusters alongside with random normalized weight
def generate_chromosome():
    weights = np.random.random(num_cluster)
    weights /= np.sum(weights)
    indexs = [random.choice(clusters[i]) for i in range(num_cluster) ]

    return list(zip(indexs,weights))



class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=2,
                         n_constr=0,
                         elementwise_evaluation=False)

    def _evaluate(self, x, out, *args, **kwargs):
        t = x.tolist()
        f = []
        tempChromosomes.clear()
        tempObjectives.clear()
        for i in range(len(t)):
            portfolio = t[i][0].tolist()
            for j in range(len(portfolio)):
                portfolio[j][0] = int(portfolio[j][0])
            #     portfolio[j][1] = float(portfolio[j][1])
            p,w = zip(*portfolio)
            result = eval(list(p),np.array(w))
            f.append(result)
            tempChromosomes.append(t[i])
            tempObjectives.append(result)
        out["F"] = np.array(f)
        

class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=np.object)
        list = np.array([])
        for i in range(n_samples):
            while True:
                chromosome = generate_chromosome()
                r = [(gene == chromosome).all() for gene in list]
                if np.array(r).all() == False or len(list) == 0:
                    break
            np.append(list,chromosome)
            
            X[i, 0] = np.array(chromosome)
        return X


def findAvaliableParent(X,k,maxRisk,n_matings):
    # get the first and the second parents
    a = X[0, k, 0]
    b = X[1, k, 0]
    if algorithm_name == 'MOEAD':
        return a, b
    start = time.time()
    # Limit risk.
    # risk <= max(rThreshold, maxRisk* rFactor)
    riskA = 0
    riskB = 0
    for i in range(len(tempChromosomes)):
        if riskA != 0 and riskB !=0:
            break
        if (tempChromosomes[i][0] == a).all():
            riskA = tempObjectives[i][1]
        if (tempChromosomes[i][0] == b).all():
            riskB = tempObjectives[i][1]
    
    while riskA >= max(rThreshold,maxRisk * rFactor) or riskB >= max(rThreshold,maxRisk * rFactor):
        riskA = 0
        riskB = 0
        newK = random.randint(0,n_matings-1)
        a = X[0,newK,0]
        b = X[1,newK,0]
        for i in range(len(tempChromosomes)):
            if riskA != 0 and riskB !=0:
                break
            if (tempChromosomes[i][0] == a).all():
                riskA = tempObjectives[i][1]
            if (tempChromosomes[i][0] == b).all():
                riskB = tempObjectives[i][1]
        if time.time() - start > 1:
            break
    return a, b
class MyCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):


        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=np.object)

        # maxRisk in current population
        maxRisk = -1
        for i in tempObjectives:
            maxRisk = max(maxRisk,i[1])

        # for each mating provided
        for k in range(n_matings):
            a, b = findAvaliableParent(X,k,maxRisk,n_matings)
            off_a = np.copy(a)
            off_b = np.copy(b)

            # Random crossover implemented by exchanging asset alongside with weight
            for i in range(len(a)):
                if random.random() < crossover_rate:
                    off_a[i] = b[i]
                    off_b[i] = a[i]
            Y[0, k, 0], Y[1, k, 0] = off_a, off_b

        return Y


class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        # Random mutation: 1. Asset mutation 2. Weight redistribution
        for i in range(len(X)):
            # Asset mutation
            if random.random() < mutaion_rate:
                row = random.randint(0,X[0][0].shape[0]-1)
                for t in clusters:
                    if X[i][0][row][0] in t:
                        X[i][0][row][0] = random.choice(t)
            # Weight redistribution
            if random.random() < mutaion_rate:
                row1 = random.randint(0,X[i][0].shape[0]-1)
                row2 = random.randint(0,X[i][0].shape[0]-1)
                new_weight1 = random.uniform(0,X[i][0][row1][1]+X[i][0][row2][1])
                new_weight2 = X[i][0][row1][1]+X[i][0][row2][1] - new_weight1
                X[i][0][row1][1] = new_weight1
                X[i][0][row2][1] = new_weight2
            

        return X


class MyDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        la = a.X.tolist()[0].tolist()
        lb = b.X.tolist()[0].tolist()
        for i in range(len(la)):
            if la[i][0] != lb[i][0] or la[i][1] != lb[i][1]:
                    return False
        return True



def k_means(data):
    x = []
    for i in range(len(data)):
        x.append(eval([i],np.array([1])))

    kmeans = KMeans(n_clusters=num_cluster,random_state=0).fit(np.array(x))
    for i in range(num_cluster):
        clusters.append(np.where(kmeans.labels_==i)[0].tolist())

def data_file_reader(file_name):
    data = []
    symbols = []
    with open(file_name, newline='\n') as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            if row[1] not in symbols:
                symbols.append(row[1])
                data.append([])
            data[symbols.index(row[1])].append(float(row[3]))

    return data,symbols

def MOEA_algorithm(algorithm_name):
    if algorithm_name == 'NSGA2':
        return NSGA2(pop_size=p_size, sampling=MySampling(), crossover=MyCrossover(), mutation=MyMutation(), eliminate_duplicates= MyDuplicateElimination())
    if algorithm_name == 'NSGA3':
        return NSGA3(pop_size=p_size, ref_dirs=get_reference_directions('das-dennis', 2, n_partitions=n_partitions),sampling=MySampling(), crossover=MyCrossover(), mutation=MyMutation(), eliminate_duplicates= MyDuplicateElimination())
    if algorithm_name == 'MOEAD':
        return MOEAD(get_reference_directions("das-dennis",2,n_partitions=p_size-1),n_neighbors=n_neighbors,decomposition=decomposition,sampling=MySampling(),crossover=MyCrossover(),mutation=MyMutation(),eliminate_duplicates=MyDuplicateElimination())
    if algorithm_name == 'CTAEA':
        return CTAEA(ref_dirs=get_reference_directions('das-dennis', 2, n_partitions=p_size-1),sampling=MySampling(), crossover=MyCrossover(), mutation=MyMutation(), eliminate_duplicates= MyDuplicateElimination())
def visualization():
    return 2





def infoDisplay():
    # Basic info display
    info = "Multi-Objective Evolutionary Portfolio Optimization\nNumber of cluster:{num_cluster}\nTime interval:{interval}\nAlgorithm:{algorithm_name}\nPopulation size:{p_size}\nNumber of generation:{n_gen}\nCrossover rate:{crossover_rate}\nMutation rate:{mutation_rate}\n"
    MOEAD_addtition_info = "Decomposition approach:{decomposition}\nPartition number of reference direction:{n_partitions}\nNumber of neighbors:{n_neighbors}\n"
    reference_direction_info = "Partition number of reference direction:{n_partitions}"
    if algorithm_name == "NSGA2":
        print(info.format(num_cluster=num_cluster,interval=interval,algorithm_name=algorithm_name,p_size=p_size,n_gen=n_gen,crossover_rate=crossover_rate,mutation_rate=mutaion_rate))
    if algorithm_name == "NSGA3":
        info += reference_direction_info
        print(info.format(num_cluster=num_cluster,interval=interval,algorithm_name=algorithm_name,p_size=p_size,n_gen=n_gen,crossover_rate=crossover_rate,mutation_rate=mutaion_rate,n_partitions=n_partitions))
    if algorithm_name == "MOEAD" or algorithm_name == "CTAEA":
        info += MOEAD_addtition_info
        print(info.format(num_cluster=num_cluster,interval=interval,algorithm_name=algorithm_name,p_size=n_partitions+1,n_gen=n_gen,crossover_rate=crossover_rate,mutation_rate=mutaion_rate,decomposition=decomposition,n_partitions=n_partitions,n_neighbors=n_neighbors))

def pfTuning(pf,chromosome):
    new_pf = []
    new_chromosome = []
    for i in range(len(pf)):
        if(pf[i][1]<rThreshold):
            new_pf.append(pf[i])
            new_chromosome.append(chromosome[i])
    return np.array(new_pf),np.array(new_chromosome)

def run():
    problem = MyProblem()
    algorithm = MOEA_algorithm(algorithm_name)
    obj = copy.deepcopy(algorithm)
    obj.setup(problem,("n_gen",n_gen),verbose=True, seed=1)
    pfs = []
    chromosomes = []
    while obj.has_next():
        obj.next()
        #print(f"gen: {obj.n_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV').min()} ideal: {obj.opt.get('F')}")
        #print(obj.opt.get('X').tolist())
        pfs.append(obj.opt.get('F'))
        chromosomes.append(obj.opt.get('X').tolist())
    
    return pfs, chromosomes
    
    

def runWithParametersOptimization(optimization_approach):
    if optimization_approach == "random":
        randomParametersOptimization()
    if optimization_approach == "rl":
        rlParametersOptimization()
    if optimization_approach == "ann":
        annParametersOptimization()


def randomParametersOptimization():
    pfs = []
    for _ in range(epoch):
        mutaion_rate = random.random()
        crossover_rate = random.random()
        num_cluster = random.randint(range_cluster)
        rFactor = random.random()
        while rFactor<=range_rFactor[0] or rFactor >= range_rFactor[1]:
            rFactor = random.random()
        n_partitions = random.randint(range_partitions)
        k_means(data)
        pfs.append(run()[-1])

    final_pf = pfs[0]
    for pf in pfs:
        final_pf = pfMeasureMetric(pf,final_pf)

    drawAndSave(final_pf)



def pfMeasureMetric(pf1,pf2):

    return 1

def drawAndSave(pf):
    plt.scatter(-pf[:,0],pf[:,1],color='blue',label='pareto front')
    plt.title('Pareto Front by ' + algorithm_name)
    plt.xlabel('Expected Annual Return(in percent scale)')
    plt.ylabel('Standard Deviation(in percent scale)')
    plt.legend()
    plt.savefig('Pareto Front by ' + algorithm_name + '.png')


def rlParametersOptimization():
    return 1

def annParametersOptimization():
    return 1
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Multi-Objective Evolutionary Portfolio Optimization')
    parser.add_argument("-a","--algorithm",help="MOEAD, NSGA2, NSGA3")
    parser.add_argument("-d","--decomposition",help="pbi, tchebi")
    parser.add_argument("-p","--population_size",type=int, help="Population size for MOEA")
    parser.add_argument("-g","--n_gen",type=int, help="Number of Generation")
    parser.add_argument("-c","--crossover_rate",type=float,help="Probability of crossover")
    parser.add_argument("-m","--mutation_rate",type=float,help="Probability of mutation")
    parser.add_argument("-s","--n_partitions",type=int,help="Partition number of reference direction")
    parser.add_argument("-n","--n_neighbors",type=int,help="Number of neighbors in MOEA/D")
    parser.add_argument("-t","--time_interval",type=int,help="Time interval for objective calculation")
    parser.add_argument("-q","--num_cluster",type=int,help="Number of clusters in kmeans")
    parser.add_argument("-o","--optimization_approach",help="Automatically optimize hyper parameters: random, rl, ann ") 
    
    args = parser.parse_args()
    if args.algorithm:
        algorithm_name = args.algorithm
    if args.decomposition:
        decomposition = args.decomposition
    if args.population_size:
        p_size = args.population_size
    if args.n_gen:
        n_gen = args.n_gen
    if args.crossover_rate:
        crossover_rate = args.crossover_rate
    if args.mutation_rate:
        mutaion_rate = args.mutaion_rate
    if args.n_partitions:
        n_partitions = args.n_partitions
    if args.n_neighbors:
        n_neighbors = args.n_neighbors
    if args.time_interval:
        interval = args.time_interval
    if args.num_cluster:
        num_cluster = args.num_cluster

    file_name = '/Users/ryan/Projects/FYP4CNN/new_prices.csv'
    data, symbols = data_file_reader(file_name)
    

    if args.optimization_approach:
        runWithParametersOptimization(args.optimization_approach)
    else:
        k_means(data)
        infoDisplay()
        pfs,chromosomes = run()
        pf,chromosome = pfTuning(pfs[-1],chromosomes[-1])
        drawAndSave(pf)