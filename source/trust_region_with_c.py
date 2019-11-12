import pandas as pd
import numpy as np
import scipy.optimize
import ctypes

def enumerable2ctypes(enumerable):
    t = ctypes.c_double*len(enumerable)
    arr = t()
    for i, value in enumerate(enumerable):
        arr[i] = value
    arr_len = ctypes.c_int
    arr_len = len(enumerable)
    return arr, arr_len
def matrix2ctypes(enumerable):
    t = ctypes.c_double*len(enumerable)
    arr = t()
    for i, value in enumerate(enumerable):
        arr[i] = value
    arr_height = ctypes.c_int
    arr_width = ctypes.c_int
    arr_height = len(enumerable)
    arr_width = enumerable.strides[0]
    return arr, arr_height, arr_width

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

args_arr, args_len = enumerable2ctypes(args)
initial_state_arr, initial_state_len = enumerable2ctypes(initial_state.values)
N_arr, N_len = enumerable2ctypes(population_df.values)
Ibar_arr, Ibar_height, Ibar_width = matrix2ctypes(I_df.values)






def get_beta(args):
    bias = arg_sizes[0] + arg_sizes[1]
    return args[bias]
def get_gamma(args):
    bias = arg_sizes[0]
    return args[bias+0: bias+size]
def get_A(args, i, j):
    return args[size*i+j]

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
        result = N(i) - I(i, t, args)
    else:
        result = -summed(i, t-1, args)*S(i, t-1, args) + S(i, t-1, args)
    R_results[(i, t)] = result
    return result
def summed(i, t, args):
    if (i, t) in summed_results:
        return summed_results[(i, t)]
    result = 0
    for j in range(0, size):
        result += get_A(args, i, j)*I(j, t, args)/N(j)
    summed_results[(i, t)] = result
    return result
def N(i):
    state_name = state_map_dict[i]
    return population_df[state_name]
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
            result += 0.5* (I(i, t, args)-I_bar(i, t))**2
    dict_clear()
    return result
'''
while True:
    print(f(args))
    args = np.random.rand(total_size)
'''
#scipy.optimize.minimize(f, x0=args, method='trust-ncg', jac='2-point', hess='2-point')
scipy.optimize.minimize(f, x0=args, options={'disp':True})
print('!')
