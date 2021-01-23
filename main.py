
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
#assets = ["GOOG", "AAPL", "TSLA", "AMZN", "NFLX"]
#assets = { 1 : "GOOG", 2 : "AAPL", 3 : "TSLA", 4 : "AMZN", 5 : "NFLX"}


def eval(portfolio,weights):

    return np.array([random.randint(1,100),random.randint(1000,100000)])
    #Get the stock starting date
    stockStartDate = '2019-01-01'
    # Get the stocks ending date aka todays date and format it in the form YYYY-MM-DD
    today = datetime.today().strftime('%Y-%m-%d')

    #Create a dataframe to store the adjusted close price of the stocks
    df = pd.DataFrame()
    #Store the adjusted close price of stock into the data frame

    for stock in portfolio:
       df[stock] = web.DataReader(stock,data_source='yahoo',start=stockStartDate , end=today)['Close']
    df

    # Create the title 'Portfolio Close Price History'
    title = 'Portfolio Close Price History    '
    #Get the stocks
    my_stocks = df
    #Create and plot the graph
    plt.figure(figsize=(12.2,4.5)) #width = 12.2in, height = 4.5
    # Loop through each stock and plot the Adj Close for each day
    for c in my_stocks.columns.values:
        plt.plot( my_stocks[c],  label=c)#plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
        plt.title(title)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price USD ($)',fontsize=18)
        plt.legend(my_stocks.columns.values, loc='upper left')

    #Show the daily simple returns, Formula = new_price/old_price - 1
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 21
    #Expected portfolio variance= WT * (Covariance Matrix) * W
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    port_standardD = np.sqrt(port_variance)
    portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights) * 21

    percent_var = round(port_variance, 2) * 100
    percent_std = round(port_standardD, 2) * 100
    percent_ret = round(portfolioSimpleAnnualReturn, 2)*100
    
#    print("Expected annual return : "+ percent_ret)
#    print('Annual volatility/standard deviation/risk : '+percent_std)
#    print('Annual variance : '+percent_var)
    
    return np.array([percent_ret, percent_std])
    


search_space = ["GOOG", "AAPL", "TSLA", "AMZN", "NFLX"]


def generate_chromosome(xl,xu):
    num = random.randint(xl, xu)
    np.random.shuffle(search_space)
    weights = np.random.random(num)
    weights /= np.sum(weights)
#    pdb.set_trace()
    return np.concatenate((search_space[:num], weights),axis=0)


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=2,
                         n_constr=0,
                         elementwise_evaluation=False)

    def _evaluate(self, x, out, *args, **kwargs):
        
        q = x.tolist()
        f = []
#        pdb.set_trace()
        for item in q:
            arr = item[0].tolist()
            portfolio = arr[:len(arr)//2]
            weights = arr[len(arr)//2:]
            
            for i in range(len(weights)):
                try:
                    weights[i] = float(weights[i])
                except ValueError:
                    print(weights[i])
                    pdb.set_trace()
            result = eval(portfolio,np.array(weights))
            f.append(result)

        out["F"] = np.array(f)
        

class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=np.object)
        
        list = np.array([])
        for i in range(n_samples):

            while True:
                chromosome = generate_chromosome(1,4)
                r = [(gene == chromosome).all() for gene in list]

                if np.array(r).all() == False or len(list) == 0:
                    break

            np.append(list,chromosome)
            
            X[i, 0] = np.array(chromosome, dtype='<U18')
        
        return X


class MyCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        #        pdb.set_trace()

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
            a, b = X[0, k, 0], X[1, k, 0]
        
            
            
            
            
            cut = random.randint(1,min(len(a)//2,len(b)//2))
            
            off_a = np.concatenate((a[:cut], b[cut:len(b)//2], a[len(a)//2:len(a)//2+cut], b[len(b)//2+cut:]), axis=0)
            
            off_b = np.concatenate((b[:cut], a[cut:len(a)//2], b[len(b)//2:len(b)//2+cut], a[len(a)//2+cut:]), axis=0)
            
            
        
            Y[0, k, 0], Y[1, k, 0] = off_a, off_b

        return Y


class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        
        for i in range(len(X)):

            r = np.random.random()

            # with a probabilty of 40% - change the order of characters
            if r < 0.2:
            
                index = random.randint(0, len(X[i].tolist()[0])//2)
                X[i][0][index] = random.choice(np.setdiff1d(np.array(search_space), X[i][0][:len(X[i][0])//2]))

            
            
        return X


class MyDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        la = a.X.tolist()[0].tolist()
        lb = b.X.tolist()[0].tolist()
        if len(la) != len(lb):
            return False
        else:
            for i in range(len(la)):
                if la[i] != lb[i]:
                    return False
                    
        return True




if __name__ == "__main__":
    problem = MyProblem()

    algorithm = NSGA2(pop_size=200,
                          sampling=MySampling(),
                          crossover=MyCrossover(),
                          mutation=MyMutation(),
                          eliminate_duplicates=MyDuplicateElimination())

    #     algorithm = MOEAD(get_reference_directions("das-dennis",2,n_partitions=5),n_neighbors=3,decomposition="pbi",pop_size=16,sampling=MySampling(),crossover=MyCrossover(),mutation=MyMutation(),eliminate_duplicates=MyDuplicateElimination())
    # algorithm = NSGA3(pop_size=200, ref_dirs=get_reference_directions("das-dennis", 2, n_partitions=150),
    #                   sampling=MySampling(), crossover=MyCrossover(), mutation=MyMutation(),
    #                   eliminate_duplicates=MyDuplicateElimination())
    obj = copy.deepcopy(algorithm)
    obj.setup(problem, ("n_gen", 50), verbose=True, seed=1)
    while obj.has_next():
        obj.next()
        print(f"gen: {obj.n_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV').min()} ideal: {obj.opt.get('F')}")
        print(obj.opt.get('X').tolist())
    result = obj.result()
