
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
import torch.multiprocessing as mp
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.algorithms.moead import MOEAD
from pymoo.algorithms.nsga3 import NSGA3
import copy
from sklearn.cluster import KMeans
import csv
import os
import argparse


# Model parameters
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

# MOEA parameters
# Probability of crossover
crossover_rate = 0.5
# Probability of mutation
mutaion_rate = 0.5
# Population size
p_size = 200
# Number of generation
n_gen = 50
# Partition number of reference direction
n_partitions = 150
# MOEA/D decomposition approach. Support 'pbi' and 'tchebi'
decomposition = 'tchebi'
# Number of neighbor in MOEA/D
n_neighbors = 20
# MOEA algorithm
algorithm_name = 'NSGA2'



def eval(portfolio,weights):


    # Online data method
    # #Get the stock starting date
    # stockStartDate = '2019-01-01'
    # # Get the stocks ending date aka todays date and format it in the form YYYY-MM-DD
    # today = datetime.today().strftime('%Y-%m-%d')

    # #Create a dataframe to store the adjusted close price of the stocks
    # df = pd.DataFrame()
    # #Store the adjusted close price of stock into the data frame

    # for stock in portfolio:
    #    df[stock] = web.DataReader(stock,data_source='yahoo',start=stockStartDate , end=today)['Close']

    # Sum for each time interval
    df = {}
    for asset in portfolio:
        temp = np.array(data[asset])
        temp_sum = [np.sum(temp[i*interval:(i+1)*interval]) for i in range(len(data[asset])//interval)]
        df[asset] = temp_sum
    df = pd.DataFrame(df)


    # Create the title 'Portfolio Close Price History'
    # title = 'Portfolio Close Price History    '
    #Get the stocks
    my_stocks = df

    # #Create and plot the graph
    # plt.figure(figsize=(12.2,4.5)) #width = 12.2in, height = 4.5
    # # Loop through each stock and plot the Adj Close for each day
    # for c in my_stocks.columns.values:
    #     plt.plot( my_stocks[c],  label=c)#plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
    #     plt.title(title)
    #     plt.xlabel('Date',fontsize=18)
    #     plt.ylabel('Price USD ($)',fontsize=18)
    #     plt.legend(my_stocks.columns.values, loc='upper left')

    #Show the daily simple returns, Formula = new_price/old_price - 1
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 21
    #Expected portfolio variance= WT * (Covariance Matrix) * W
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    port_standardD = np.sqrt(port_variance)
    portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights) * 21

    percent_var = round(port_variance, 8) * 100
    percent_std = round(port_standardD, 8) * 100
    percent_ret = round(portfolioSimpleAnnualReturn, 8)*100
    
#    print("Expected annual return : "+ percent_ret)
#    print('Annual volatility/standard deviation/risk : '+percent_std)
#    print('Annual variance : '+percent_var)
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
        for i in range(len(t)):
            portfolio = t[i][0].tolist()
            for j in range(len(portfolio)):
                portfolio[j][0] = int(portfolio[j][0])
            #     portfolio[j][1] = float(portfolio[j][1])
            p,w = zip(*portfolio)
            result = eval(list(p),np.array(w))
            f.append(result)
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
        #        pdb.set_trace()
        # for each mating provided
        for k in range(n_matings):
            #            pdb.set_trace()
            # get the first and the second parent
            a = X[0, k, 0]
            b = X[1, k, 0]
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
    if algorithm_name == 'NAGA2':
        return NSGA2(pop_size=p_size, sampling=MySampling(), crossover=MyCrossover(), mutation=MyMutation(), eliminate_duplicates= MyDuplicateElimination())
    if algorithm_name == 'NSGA3':
        return NSGA3(pop_size=p_size, ref_dirs=get_reference_directions('das-dennis', 2, n_partitions=n_partitions),sampling=MySampling(), crossover=MyCrossover(), mutation=MyMutation(), eliminate_duplicates= MyDuplicateElimination())
    if algorithm_name == 'MOEAD':
        return MOEAD(get_reference_directions("das-dennis",2,n_partitions=n_partitions),n_neighbors=n_neighbors,decomposition=decomposition,pop_size=p_size,sampling=MySampling(),crossover=MyCrossover(),mutation=MyMutation(),eliminate_duplicates=MyDuplicateElimination())

if __name__ == "__main__":
    
    file_name = '/Users/ryan/Projects/FYP4CNN/new_prices.csv'

    data, symbols = data_file_reader(file_name)
    k_means(data)

    problem = MyProblem()

    # algorithm = NSGA2(pop_size=200,
    #                       sampling=MySampling(),
    #                       crossover=MyCrossover(),
    #                       mutation=MyMutation(),
    #                       eliminate_duplicates=MyDuplicateElimination())

    # #     algorithm = MOEAD(get_reference_directions("das-dennis",2,n_partitions=5),n_neighbors=3,decomposition="pbi",pop_size=16,sampling=MySampling(),crossover=MyCrossover(),mutation=MyMutation(),eliminate_duplicates=MyDuplicateElimination())
    # # algorithm = NSGA3(pop_size=200, ref_dirs=get_reference_directions("das-dennis", 2, n_partitions=150),
    # #                   sampling=MySampling(), crossover=MyCrossover(), mutation=MyMutation(),
    # #                   eliminate_duplicates=MyDuplicateElimination())

    algorithm = MOEA_algorithm(algorithm_name)
    obj = copy.deepcopy(algorithm)
    obj.setup(problem, ("n_gen", 50), verbose=True, seed=1)
    while obj.has_next():
        obj.next()
        print(f"gen: {obj.n_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV').min()} ideal: {obj.opt.get('F')}")
        print(obj.opt.get('X').tolist())
    result = obj.result()
