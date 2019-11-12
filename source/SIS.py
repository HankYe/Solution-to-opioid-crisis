import pandas as pd
import numpy as np
import scipy.optimize
import numdifftools as nd
from pyswarm import pso
from matplotlib import pyplot
import pickle
import time

size = 5
train_time = 8
max_time = 8

state_map_dict = {0:'KY', 1:'OH', 2:'PA', 3:'VA', 4:'WV'}
time_map_dict = {0:2010, 1:2011, 2:2012, 3:2013, 4:2014, 5:2015, 6:2016, 7:2017}
time_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
full2abbrev_dict = {'Kentucky':'KY', 'Ohio':'OH', 'Pennsylvania':'PA', 'Virginia':'VA', 'West Virginia':'WV'} 

I_df = pd.read_csv('MCM_NFLIS_Data.csv')
I_df = I_df.groupby(['State', 'YYYY'])['DrugReports'].sum()
I_dict = {}
for i in range(0, size):
    for t in range(0, max_time):
        I_dict[(i, t)] = I_df[state_map_dict[i], time_map_dict[t]]

population_df = pd.read_csv('ACS_10_5YR_DP02_with_ann.csv')
population_df = population_df.iloc[1:]
population_df['HC01_VC128'] = population_df['HC01_VC128'].apply(lambda x:int(x))
population_df['State'] = population_df['GEO.display-label'].apply(lambda x:full2abbrev_dict[x.split(', ')[1]])
population_df = population_df.groupby(['State'])['HC01_VC128'].sum()


initial_state = I_df[I_df.index.map(lambda x:x[1])==2010]
mediate_state = I_df[I_df.index.map(lambda x:x[1])==2017]
'''
gamma = np.random.rand(size)
beta = np.random.rand()
A = np.random.rand(size, size)
'''
arg_sizes = [size*size, size, size]
total_size = sum(arg_sizes)
args = np.random.rand(total_size)
bounds = []
lb = []
ub = []
bias = 0
for i in range(0, size):
    for j in range(0, size):
        '''
        if (i+1, j+1) in not_connected or (j+1, i+1) in not_connected:
            bounds.append((0, 0))
        else:
            bounds.append((0, 0.5))
        '''
        if i == j:
            bounds.append((0.5, 2))
        else:
            bounds.append((0, 0.5))
bias += arg_sizes[0]
for i in range(bias, bias+arg_sizes[1]):#gamma_0
    bounds.append((0.2, 1))
bias += arg_sizes[1]
for i in range(bias, bias+arg_sizes[2]):#gamma_1
    bounds.append((-0.1, 0.1))

def get_gamma_1(args):
    bias = arg_sizes[0] + arg_sizes[1]
    return args[bias+0: bias+size]
def get_gamma_0(args):
    bias = arg_sizes[0]
    return args[bias+0: bias+size]
get_A = lambda args, i, j: args[size*i+j]

I_results = {}
S_results = {}
summed_results = {}

steps_per_t = 3
def I_step(i, t, step, args):
    key_tuple = (i, t, step)
    if key_tuple in I_results:
        return I_results[key_tuple]
    if (t, step) == (0, 0):
        result = initial_state[i]
    else:
        if step == 0:
            t -= 1
            step = steps_per_t
        result = I_step(i, t, step-1, args) + S_step(i, t, step-1, args) -S_step(i, t, step, args)
    I_results[key_tuple] = result
    return result
def S_step(i, t, step, args):
    key_tuple = (i, t, step)
    if key_tuple in S_results:
        return S_results[key_tuple]
    if (t, step) == (0, 0):
        result = fastN(i) - I_step(i, t, step, args)
    else:
        if step == 0:
            t -= 1
            step = steps_per_t
        gamma = get_gamma_0(args)[i] + get_gamma_1(args)[i]*(t+step/steps_per_t)
        gamma = max(min(1, gamma), 0)
        result = (-summed_step(i, t, step-1, args)*S_step(i, t, step-1, args) + gamma*I_step(i, t, step-1, args))/steps_per_t + S_step(i, t, step-1, args)

        '''
        ratio = max(1-get_gamma(args)[i]*I(i, t-1, args)/I(i, max(t-2, 0), args), 0)
        result = -summed(i, t-1, args)*S(i, t-1, args)*ratio + get_gamma(args)[i]*I(i, t-1, args) + S(i, t-1, args)
        '''
    S_results[key_tuple] = result
    return result
def summed_step(i, t, step, args):
    key_tuple = (i, t, step)
    if key_tuple in summed_results:
        return summed_results[key_tuple]
    result = 0
    for j in range(0, size):
        result += get_A(args, i, j)*I_step(j, t, step, args)/fastN(j)
    summed_results[key_tuple] = result
    return result

def I(i, t, args):
    if (i, t) in I_results:
        return I_results[(i, t)]
    if t == 0:
        state_name = state_map_dict[i]
        #result = (get_beta(args)*10) *initial_state[state_name].values[0]
        result = initial_state[state_name].values[0]
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
fastN = lambda i:population_df.values[i]
def N(i):
    state_name = state_map_dict[i]
    return population_df[state_name]
fastI_bar = lambda i, t:I_dict[(i, t)]
def I_bar(i, t):
    return I_df[state_map_dict[i], time_map_dict[t]]
def dict_clear():
    I_results.clear()
    S_results.clear()
    summed_results.clear()
def f(args):
    result = 0
    for i in range(0, size):
        for t in range(0, train_time):
            result += abs( (I_step(i, t, 0, args)-fastI_bar(i, t)) / fastI_bar(i, t) )
    result = result / (size*train_time)

    dict_clear()
    return result
def f_test(args):
    result = 0
    for i in range(0, size):
        for t in range(train_time, max_time):
            result += abs( (I_step(i, t, 0, args)-fastI_bar(i, t)) / fastI_bar(i, t) )
    result = result / (size*(max_time-train_time))

    dict_clear()
    return result
def plot(opt_args):
    time_list_append = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    for i in range(0, size):
        predict = []
        real = []
        for t in range(0, max_time):
            predict.append(I_step(i, t, 0, opt_args))
            real.append(fastI_bar(i, t))
        for t in range(max_time, len(time_list_append)):
            predict.append(I_step(i, t, 0, opt_args))
        pyplot.plot(time_list_append, predict)
        pyplot.plot(time_list, real)
        pyplot.xlabel('Year')
        pyplot.ylabel('DrugReports')
        pyplot.title(state_map_dict[i])
        pyplot.legend(['predict', 'real'])
        pyplot.show()
def plot_mediate(args, together=False, show=True, plt=None):
    if plt is None:
        plt = pyplot
    time_list_append = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    for i in range(0, size):
        predict = []
        real = []
        for t in range(0, max_time):
            real.append(fastI_bar(i, t))
        for t in range(max_time-1, len(time_list_append)):
            predict.append(I_step(i, t-max_time+1, 0, args))
        plt.plot(time_list_append[max_time-1:], predict)
        plt.plot(time_list, real)
        pyplot.xticks(time_list_append[::4])
        if show:
            plt.xlabel('Year')
            plt.ylabel('DrugReports')
        else:
            pass
            #pyplot.xlabel('gamma_0={0}, gamma_1={1}'.format(get_gamma_0(args)[0], get_gamma_1(args)[1]))
        if not together:
            plt.title(state_map_dict[i])
            plt.show()
            
    if together:
        if state is None:
            pyplot.title('All 5 States')
        else:
            if show:
                pyplot.title('Districts of {0}'.format(state))
            legend_list = []
            for i in range(0, size):
                legend_list.append('predict{0}'.format(i+1))
                legend_list.append('real{0}'.format(i+1))
            if show:
                pyplot.legend(legend_list)
        if show:
            pyplot.show()
    dict_clear()
def get_arr_A(args):
    arr = np.ndarray((size, size))
    for i in range(0, size):
        for j in range(0, size):
            arr[i][j] = get_A(args, i, j)
    return arr
'''
while True:
    start = time.time()
    print(f(args))
    args = np.random.rand(total_size)
    print(time.time()-start)
'''

result = pickle.load(open('result(state)_final', 'rb'))

#xopt, fopt = pso(f, lb, ub, maxiter=1000)
#result = scipy.optimize.differential_evolution(f, bounds, recombination=1, disp=True, maxiter=200)
#pickle.dump(result, open('result(state)', 'wb'))
#scipy.optimize.minimize(f, x0=args, method='trust-ncg', jac=np.gradient, hess=lambda x: nd.Hessian(f)(x), options={'disp':True})
#scipy.optimize.minimize(f, x0=args, options={'disp':True})
print('train error:'+str(f(result.x)))
print(get_arr_A(result.x))
print(get_gamma_0(result.x))
print(get_gamma_1(result.x))


grad = scipy.optimize.approx_fprime(result.x, f, 0.00001)
print(get_arr_A(grad))
print(get_gamma_0(grad))
print(get_gamma_1(grad))
x = result.x.copy()
for i,_ in enumerate(get_gamma_0(x)):
    get_gamma_0(x)[i] += get_gamma_1(x)[i] * max_time
    print(get_gamma_0(x)[i])
'''
for i,_ in enumerate(get_gamma_0(x)):
    get_gamma_1(x)[i] = 0
    get_gamma_0(x)[i] = 0.85
'''
get_gamma_1(x)[1] = 0
get_gamma_0(x)[1] = 0.75
for i in range(0, size):
    for j in range(0, size):
        if not i == j:
            x[size*i+j] *= 0.8

initial_state = mediate_state
#plot_mediate(x)

print('!')
