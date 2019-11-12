import pandas as pd
import numpy as np
import scipy.optimize
from matplotlib import pyplot
import pickle
import time
import math

train_time = 8
max_time = 8
max_time_append = 13

'SyntheticOpioid'
drug = 'All'


state_map_dict = {0:'KY', 1:'OH', 2:'PA', 3:'VA', 4:'WV'}
size = 5
time_map_dict = {0:2010, 1:2011, 2:2012, 3:2013, 4:2014, 5:2015, 6:2016, 7:2017}
time_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
full2abbrev_dict = {'Kentucky':'KY', 'Ohio':'OH', 'Pennsylvania':'PA', 'Virginia':'VA', 'West Virginia':'WV'} 

#def load_args_from_txt(filename, args):


I_df = pd.read_csv('MCM_NFLIS_Data.csv')
#I_df = I_df.groupby(['COUNTY', 'YYYY'])['TotalDrugReportsCounty'].mean()
if drug == 'All':
    pass
elif drug == 'SyntheticOpioid':
    I_df = I_df[I_df['SubstanceName']!='Heroin']
else:
    I_df = I_df[I_df['SubstanceName']==drug]
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
population_list = []
for i in range(0, size):
    population_list.append(int(population_df[state_map_dict[i]]))




initial_state = []
for i in range(0, size):
    initial_state.append(I_dict[(i, 0)])
mediate_state = []
for i in range(0, size):
    mediate_state.append(I_dict[(i, max_time-1)])



l_size = 3
k_size = [5, 7, 2]
l_k_total_size = sum(k_size)

bounds = []
vulnerables = {}
pos_tuple = (0, 50)
neg_tuple = (-50, 0)
neutral_tuple = (-20, 20)
factor_dict = {(0, 0):pos_tuple, (0, 1):neg_tuple, (0, 2):pos_tuple, (0, 3):pos_tuple, (0, 4):pos_tuple,
(1, 0):neutral_tuple, (1, 1):neutral_tuple, (1, 2):neutral_tuple, (1, 3):neutral_tuple, (1, 4):neg_tuple, (1, 5):neg_tuple, (1, 6):neg_tuple,
(2, 0):neutral_tuple, (2, 1):neutral_tuple}  #factor_dict[(l, k)]
for i in range(0, size):
    for l in range(0, l_size):
        for k in range(0, k_size[l]):
            bounds.append(factor_dict[(l, k)])

linear_reg_dict = {}
def load_reg_dict():
    f = open('regression.txt', 'r')
    for l in range(0, l_size):
        for k in range(0, k_size[l]):
            for i in range(0, size):
                linear_reg_dict[(i, l, k)] = eval('('+f.readline()+')')
        for i in range(0, size):
            linear_reg_dict[(i, l)] = eval('('+f.readline()+')')

load_reg_dict()
def get_rho(i, l, k, rhos):
    return rhos[i*l_k_total_size + sum(k_size[:l]) + k]
def print_rho(rhos):
    for l in range(0, l_size):
        for i in range(0, size):
            print('l={0}, i={1}'.format(l, i))
            for k in range(0, k_size[l]):
                print('     '+str(get_rho(i, l, k, rhos)))
def factor_predict(i, l, k, t):
    if k is None:
        a,b = linear_reg_dict[(i, l)]
    else:
        a,b = linear_reg_dict[(i, l, k)]
    return b + a*t
def factor_ratio(i, l, k, t):
    return factor_predict(i, l, k, t)/factor_predict(i, l, None, t)
def vulnerable(i, t, rhos):
    result = 0
    for l in range(0, l_size):
        for k in range(0, k_size[l]):
            result += factor_ratio(i, l, k, t) * get_rho(i, l, k, rhos)
    result = result / l_k_total_size
    result = math.exp(result)
    return result
def init_vulnerables(rhos):
    for i in range(0, size):
        for t in range(0, max_time_append):
            vulnerables[(i, t)] = vulnerable(i, t, rhos)

arg_sizes = [size*size, size, size]
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
        result = (-summed_step(i, t, step-1, args)*S_step(i, t, step-1, args)*vulnerables[(i, t)] + gamma*I_step(i, t, step-1, args))/steps_per_t + S_step(i, t, step-1, args)
        #result = (-summed_step(i, t, step-1, args)*S_step(i, t, step-1, args) + gamma*I_step(i, t, step-1, args))/steps_per_t + S_step(i, t, step-1, args)

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

fastN = lambda i:population_list[i]
fastI_bar = lambda i, t:I_dict[(i, t)]
def dict_clear():
    I_results.clear()
    S_results.clear()
    summed_results.clear()
def f(rhos, args):
    if len(args) == 1:
        args = args[0]
    init_vulnerables(rhos)
    result = 0
    for i in range(0, size):
        for t in range(0, train_time):
            if fastI_bar(i, t) != 0:
                result += abs( (I_step(i, t, 0, args)-fastI_bar(i, t)) / fastI_bar(i, t) )
    result = result / (size*train_time)
    vulnerables.clear()
    dict_clear()
    return result
def plot(rhos, args):
    init_vulnerables(rhos)
    time_list_append = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    for i in range(0, size):
        predict = []
        real = []
        for t in range(0, max_time):
            predict.append(I_step(i, t, 0, args))
            real.append(fastI_bar(i, t))
        for t in range(max_time, len(time_list_append)):
            predict.append(I_step(i, t, 0, args))
        pyplot.plot(time_list_append, predict)
        pyplot.plot(time_list, real)
        pyplot.xticks(time_list_append[::3])
        pyplot.xlabel('Year')
        pyplot.ylabel('{0}DrugReports'.format(drug))
        pyplot.title(state_map_dict[i])
        pyplot.legend(['predict', 'real'])
        pyplot.show()
    vulnerables.clear()
    dict_clear()
'''
def plot_mediate(args):
    time_list_append = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
    for i in range(0, size):
        predict = []
        real = []
        for t in range(0, max_time):
            real.append(fastI_bar(i, t))
        for t in range(max_time-1, len(time_list_append)):
            predict.append(I_step(i, t-max_time+1, 0, args))
        pyplot.plot(time_list_append[max_time-1:], predict)
        pyplot.plot(time_list, real)
        pyplot.xticks(time_list_append[::3])
        pyplot.xlabel('Year')
        pyplot.ylabel('{0}DrugReports'.format(drug))
        pyplot.title(state_map_dict[i])
        pyplot.legend(['predict', 'real'])
        pyplot.show()
    dict_clear()
'''
def get_arr_A(args):
    arr = np.ndarray((size, size))
    for i in range(0, size):
        for j in range(0, size):
            arr[i][j] = get_A(args, i, j)
    return arr

'''
while True:
    start = time.time()
    args = np.random.rand(total_size)
    print(f(args))
    print(time.time()-start)
'''

args_result = pickle.load(open('result(state)_final', 'rb'))
args = args_result.x

#result = scipy.optimize.differential_evolution(f, bounds, args=[args], recombination=1, disp=True, maxiter=200)
#pickle.dump(result, open('result(factors)', 'wb'))

result = pickle.load(open('result(factors)', 'rb'))

print('train error:'+str(f(result.x, args)))
#print('test error:'+str(f_test(result.x)))
print(get_arr_A(args))
print(get_gamma_0(args))
print(get_gamma_1(args))
print_rho(result.x)

plot(result.x, args)


'''
x = result.x.copy()
for i,_ in enumerate(get_gamma(x)):
    get_gamma(x)[i] = 1


initial_state = mediate_state
plot_mediate(x)



print('!')

print(get_arr_A(x))
print(get_gamma(x))
'''