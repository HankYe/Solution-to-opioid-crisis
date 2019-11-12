import pandas as pd
import numpy as np
import scipy.optimize
import numdifftools as nd
from pyswarm import pso
from matplotlib import pyplot
import pickle
import time

size = 7
train_time = 7
max_time = 8

state_map_dict = {0:'KY', 1:'OH', 2:'PA', 3:'VA', 4:'WV'}
county_map_dict = {0:'NELSON', 1:'AUGUSTA', 2:'ROCKBRIDGE', 3:'AMHERST', 4:'APPOMATTOX', 5:'BUCKINGHAM', 6:'NELSON'}
#county_map_dict = {0:'NELSON', 1:'AUGUSTA', 2:'AMHERST', 3:'BUCKINGHAM', 4:'NELSON'}
time_map_dict = {0:2010, 1:2011, 2:2012, 3:2013, 4:2014, 5:2015, 6:2016, 7:2017}
time_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
full2abbrev_dict = {'Kentucky':'KY', 'Ohio':'OH', 'Pennsylvania':'PA', 'Virginia':'VA', 'West Virginia':'WV'} 

'''
df = pd.read_csv('MCM_NFLIS_Data.csv')
df = df[df['State']=='VA']
df = df.groupby(['COUNTY', 'YYYY'])['TotalDrugReportsCounty'].mean()
total_set = set(df.index.map(lambda x:x[0]))
for year in time_list:
    d = df[df.index.map(lambda x:x[1])==year]
    counties = d.index.map(lambda x:x[0])
    total_set = total_set.intersection(set(counties))
print(total_set)
'''


I_df = pd.read_csv('MCM_NFLIS_Data.csv')
I_df = I_df[I_df['State']=='VA']
#I_df = I_df.groupby(['COUNTY', 'YYYY'])['TotalDrugReportsCounty'].mean()
I_df = I_df.groupby(['COUNTY', 'YYYY'])['TotalDrugReportsCounty'].mean()
I_dict = {}
for i in range(0, size):
    for t in range(0, max_time):
        I_dict[(i, t)] = I_df[county_map_dict[i], time_map_dict[t]]

population_df = pd.read_csv('ACS_10_5YR_DP02_with_ann.csv')
population_df = population_df.iloc[1:]
population_df['HC01_VC128'] = population_df['HC01_VC128'].apply(lambda x:int(x))
population_df['State'] = population_df['GEO.display-label'].apply(lambda x:full2abbrev_dict[x.split(', ')[1]])
population_df = population_df[population_df['State']=='VA']
population_df['COUNTY'] = population_df['GEO.display-label'].apply(lambda x:x.split(' County, ')[0].upper())
population_df = population_df.set_index('COUNTY')
population_df = population_df[['HC01_VC128']]
population_list = []
for i in range(0, size):
    population_list.append(int(population_df.loc[county_map_dict[i]].values))



initial_state = I_df[I_df.index.map(lambda x:x[1])==2010]
initial_state.index = initial_state.index.map(lambda x:x[0])
'''
gamma = np.random.rand(size)
beta = np.random.rand()
A = np.random.rand(size, size)
'''
arg_sizes = [size*size, size]
total_size = sum(arg_sizes)
args = np.random.rand(total_size)
bounds = []
lb = []
ub = []
bias = 0
for i in range(0, arg_sizes[0]):
    lb.append(-1)
    ub.append(1)
    bounds.append((-0.1, 0.1))
bias += arg_sizes[0]
for i in range(bias, bias+arg_sizes[1]):
    lb.append(0)
    ub.append(0.25)
    bounds.append((0, 0.25))
bias += arg_sizes[1]

def get_gamma(args):
    bias = arg_sizes[0]
    return args[bias+0: bias+size]
get_A = lambda args, i, j: args[size*i+j]

I_results = {}
S_results = {}
summed_results = {}
def I(i, t, args):
    if (i, t) in I_results:
        return I_results[(i, t)]
    if t == 0:
        county_name = county_map_dict[i]
        result = initial_state[county_name]
    else:
        result = I(i, t-1, args) + S(i, t-1, args) -S(i, t, args)
    I_results[(i, t)] = result
    return result
def S(i, t, args):
    if (i, t) in S_results:
        return S_results[(i, t)]
    if t == 0:
        result = fastN(i) - I(i, t, args)
    else:
        result = -summed(i, t-1, args)*S(i, t-1, args) + get_gamma(args)[i]*I(i, t-1, args) + S(i, t-1, args)
    S_results[(i, t)] = result
    return result
def summed(i, t, args):
    if (i, t) in summed_results:
        return summed_results[(i, t)]
    result = 0
    for j in range(0, size):
        result += get_A(args, i, j)*I(j, t, args)/fastN(j)
    summed_results[(i, t)] = result
    return result
fastN = lambda i:population_list[i]
def N(i):
    county_name = county_map_dict[i]
    return population_df[county_name]
fastI_bar = lambda i, t:I_dict[(i, t)]
def I_bar(i, t):
    return I_df[county_map_dict[i], time_map_dict[t]]
def dict_clear():
    I_results.clear()
    S_results.clear()
    summed_results.clear()
def f(args):
    result = 0
    for i in range(0, size):
        for t in range(0, train_time):
            result += abs( (I(i, t, args)-fastI_bar(i, t)) / fastI_bar(i, t) )
    result = result / (size*train_time)

    dict_clear()
    return result
def f_test(args):
    result = 0
    for i in range(0, size):
        for t in range(train_time, max_time):
            result += abs( (I(i, t, args)-fastI_bar(i, t)) / fastI_bar(i, t) )
    result = result / (size*(max_time-train_time))

    dict_clear()
    return result
def inspect():
    for i in range(0, size):
        for t in range(0, max_time):
            print('predict:'+str(I(i, t, args)))
            print('real:'+str(fastI_bar(i, t)))
            print('')
def plot(opt_args):
    for i in range(0, size):
        predict = []
        real = []
        for t in range(0, max_time):
            predict.append(I(i, t, opt_args))
            real.append(fastI_bar(i, t))
        pyplot.plot(time_list, predict)
        pyplot.plot(time_list, real)
        pyplot.xlabel('Year')
        pyplot.ylabel('TotalDrugReportsCounty')
        pyplot.title(county_map_dict[i])
        pyplot.legend(['predict', 'real'])
        pyplot.show()

'''
while True:
    start = time.time()
    print(f(args))
    args = np.random.rand(total_size)
    print(time.time()-start)
'''


result = scipy.optimize.differential_evolution(f, bounds, recombination=1, disp=True, maxiter=100)
pickle.dump(result, open('result_county', 'wb'))

print('!')