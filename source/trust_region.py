import pandas as pd
import numpy as np
import scipy.optimize
import numdifftools as nd
from pyswarm import pso
import time

state_map_dict = {0:'KY', 1:'OH', 2:'PA', 3:'VA', 4:'WV'}
time_map_dict = {0:2010, 1:2011, 2:2012, 3:2013, 4:2014, 5:2015, 6:2016, 7:2017}
full2abbrev_dict = {'Kentucky':'KY', 'Ohio':'OH', 'Pennsylvania':'PA', 'Virginia':'VA', 'West Virginia':'WV'} 

I_df = pd.read_csv('MCM_NFLIS_Data.csv')
I_df = I_df.groupby(['State', 'YYYY'])['TotalDrugReportsState'].mean()

population_df = pd.read_csv('ACS_10_5YR_DP02_with_ann.csv')
population_df = population_df.iloc[1:]
population_df['HC01_VC128'] = population_df['HC01_VC128'].apply(lambda x:int(x))
population_df['State'] = population_df['GEO.display-label'].apply(lambda x:full2abbrev_dict[x.split(', ')[1]])
population_df = population_df.groupby(['State'])['HC01_VC128'].sum()

size = 5
max_time = 8
initial_state = I_df[I_df.index.map(lambda x:x[1])==2010]
'''
gamma = np.random.rand(size)
beta = np.random.rand()
A = np.random.rand(size, size)
'''
arg_sizes = [size*size, size, 1]
total_size = sum(arg_sizes)
args = np.random.rand(total_size)
bounds = []
lb = []
ub = []
bias = 0
for i in range(0, arg_sizes[0]):
    lb.append(-0.5)
    ub.append(0.5)
    bounds.append((-0.5, 0.5))
bias += arg_sizes[0]
for i in range(bias, bias+arg_sizes[1]):
    lb.append(0)
    ub.append(1)
    bounds.append((0, 1))
bias += arg_sizes[1]
for i in range(bias, bias+arg_sizes[2]):
    lb.append(0.1)
    ub.append(100)
    bounds.append((0.1, 100))

def get_beta(args):
    bias = arg_sizes[0] + arg_sizes[1]
    return args[bias]
def get_gamma(args):
    bias = arg_sizes[0]
    return args[bias+0: bias+size]
get_A = lambda args, i, j: args[size*i+j]

I_results = {}
R_results = {}
S_results = {}
summed_results = {}
def I(i, t, args):
    if (i, t) in I_results:
        return I_results[(i, t)]
    if t == 0:
        state_name = state_map_dict[i]
        result = (get_beta(args)*10) *initial_state[state_name].values[0]
    else:
        result = I(i, t-1, args) + R(i, t-1, args) - R(i, t, args) + S(i, t-1, args) -S(i, t, args)
    I_results[(i, t)] = result
    return result
def R(i, t, args):
    if (i, t) in R_results:
        return R_results[(i, t)]
    if t == 0:
        result = 0
    else:   
        result = get_gamma(args)[i]*I(i, t-1, args) + R(i, t-1, args)
    R_results[(i, t)] = result
    return result
def S(i, t, args):
    if (i, t) in S_results:
        return S_results[(i, t)]
    if t == 0:
        result = fastN(i) - I(i, t, args)
    else:
        result = -summed(i, t-1, args)*S(i, t-1, args) + S(i, t-1, args)
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
fastN = lambda i:population_df.values[i]
def N(i):
    state_name = state_map_dict[i]
    return population_df[state_name]
fastI_bar = lambda it:I_df.iloc[it[0]*I_df.strides[0] + it[1]]
def I_bar(i, t):
    return I_df[state_map_dict[i], time_map_dict[t]]
def dict_clear():
    I_results.clear()
    R_results.clear()
    S_results.clear()
    summed_results.clear()
def f(args):
    result = 0
    for i in range(0, size):
        for t in range(0, max_time):
            result += (I(i, t, args)-fastI_bar((i, t))) **2
    result = result / (size*max_time)
    dict_clear()
    return result

'''
while True:
    start = time.time()
    print(f(args))
    args = np.random.rand(total_size)
    print(time.time()-start)
'''


xopt, fopt = pso(f, lb, ub, maxiter=1000)
#scipy.optimize.differential_evolution(f, bounds, recombination=1, disp=True)
#scipy.optimize.minimize(f, x0=args, method='trust-ncg', jac=np.gradient, hess=lambda x: nd.Hessian(f)(x), options={'disp':True})
#scipy.optimize.minimize(f, x0=args, options={'disp':True})
print('!')
