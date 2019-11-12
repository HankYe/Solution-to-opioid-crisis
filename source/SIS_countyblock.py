import pandas as pd
import numpy as np
import scipy.optimize
from matplotlib import pyplot
import pickle
import time

train_time = 8
max_time = 8

state = 'WV'
'SyntheticOpioid'
drug = 'Fentanyl'

'''
not_connected_dict = {'KY':[(1, 4), (1, 5), (2, 5), (2, 6)]}
not_connected = not_connected_dict[state]
'''

state_map_dict = {0:'KY', 1:'OH', 2:'PA', 3:'VA', 4:'WV'}
block_list_dict = {'KY':[['CAMPBELL', 'KENTON', 'BOONE', 'GALLATIN', 'CARROLL', 'TRIMBLE', 'OLDHAM', 'JEFFERSON', 'BULLITT', 'SPENCER', 'SHELBY', 'HENRY', 'OWEN', 'GRANT', 'PENDLETON', 'BRACKEN', 'HARRISON', 'SCOTT', 'FRANKLIN', 'ROBERTSON', 'MASON'], ['LEWIS', 'FLEMING', 'NICHOLAS', 'BOURBON', 'CLARK', 'MONTGOMERY', 'BATH', 'ROWAN', 'CARTER', 'GREENUP', 'BOYD', 'ELLIOTT', 'MORGAN', 'MENIFEE', 'POWELL', 'ESTILL', 'LEE', 'WOLFE', 'MAGOFFIN', 'JOHNSON', 'LAWRENCE', 'MARTIN', 'FLOYD', 'KNOTT', 'BREATHITT', 'PERRY', 'LETCHER', 'PIKE'], ['FAYETTE', 'WOODFORD', 'ANDERSON', 'WASHINGTON', 'NELSON', 'MARION', 'TAYLOR', 'GREEN', 'CASEY', 'LINCOLN', 'BOYLE', 'MERCER', 'JESSAMINE', 'MADISON', 'GARRARD', 'ROCKCASTLE', 'JACKSON', 'OWSLEY', 'CLAY'], ['LAUREL', 'PULASKI', 'RUSSELL', 'ADAIR', 'METCALFE', 'MONROE', 'CUMBERLAND', 'CLINTON', 'WAYNE', 'MCCREARY', 'WHITLEY', 'KNOX', 'BELL', 'LESLIE', 'HARLAN'], ['CARLISLE', 'HICKMAN', 'FULTON', 'GRAVES', 'MARSHALL', 'CALLOWAY', 'LYON', 'TRIGG', 'CHRISTIAN', 'TODD', 'LOGAN', 'SIMPSON', 'ALLEN', 'WARREN', 'BUTLER', 'EDMONSON', 'BARREN', 'HART'], ['LARUE', 'HARDIN', 'MEADE', 'BRECKINRIDGE', 'GRAYSON', 'OHIO', 'HANCOCK', 'DAVIESS', 'MCLEAN', 'MUHLENBERG', 'HOPKINS', 'WEBSTER', 'UNION', 'HENDERSON', 'CRITTENDEN', 'LIVINGSTON', 'CALDWELL', 'MCCRACKEN', 'BALLARD']],
'OH':[['ALLEN', 'AUGLAIZE', 'DEFIANCE', 'FULTON', 'HANCOCK', 'HARDIN', 'HENRY', 'LOGAN', 'LUCAS', 'MERCER', 'PAULDING', 'PUTNAM', 'VAN WERT', 'WILLIAMS', 'WOOD'], ['ASHLAND', 'CRAWFORD', 'DELAWARE', 'ERIE', 'HURON', 'KNOX', 'LORAIN', 'MARION', 'MORROW', 'OTTAWA', 'RICHLAND', 'SANDUSKY', 'SENECA', 'UNION', 'WYANDOT'], ['ASHTABULA', 'COLUMBIANA', 'CUYAHOGA', 'GEAUGA', 'HOLMES', 'LAKE', 'MAHONING', 'MEDINA', 'PORTAGE', 'STARK', 'SUMMIT', 'TRUMBULL', 'WAYNE'], ['BROWN', 'BUTLER', 'CHAMPAIGN', 'CLARK', 'CLERMONT', 'CLINTON', 'DARKE', 'GREENE', 'HAMILTON', 'MIAMI', 'MONTGOMERY', 'PREBLE', 'SHELBY', 'WARREN'], ['ADAMS', 'ATHENS', 'FAIRFIELD', 'FAYETTE', 'FRANKLIN', 'GALLIA', 'HIGHLAND', 'HOCKING', 'JACKSON', 'LAWRENCE', 'LICKING', 'MADISON', 'MEIGS', 'PERRY', 'PICKAWAY', 'PIKE', 'ROSS', 'SCIOTO', 'VINTON'], ['BELMONT', 'CARROLL', 'COSHOCTON', 'GUERNSEY', 'HARRISON', 'JEFFERSON', 'MONROE', 'MORGAN', 'MUSKINGUM', 'NOBLE', 'TUSCARAWAS', 'WASHINGTON']],
'PA':[['ARMSTRONG', 'BUTLER', 'CAMERON', 'CLARION', 'CLEARFIELD', 'CRAWFORD', 'ELK', 'ERIE', 'FOREST', 'JEFFERSON', 'LAWRENCE', 'MCKEAN', 'MERCER', 'VENANGO', 'WARREN'], ['BEDFORD', 'CENTRE', 'CLINTON', 'COLUMBIA', 'LYCOMING', 'MONTOUR', 'NORTHUMBERLAND', 'POTTER', 'SNYDER', 'SULLIVAN', 'TIOGA', 'UNION'], ['BUCKS', 'CARBON', 'LACKAWANNA', 'LEHIGH', 'LUZERNE', 'MONROE', 'MONTGOMERY', 'NORTHAMPTON', 'PHILADELPHIA', 'PIKE', 'SUSQUEHANNA', 'WAYNE', 'WYOMING'], ['ALLEGHENY', 'BEAVER', 'BLAIR', 'BRADFORD', 'CAMBRIA', 'FAYETTE', 'FRANKLIN', 'FULTON', 'GREENE', 'HUNTINGDON', 'INDIANA', 'SOMERSET', 'WASHINGTON', 'WESTMORELAND'], ['ADAMS', 'BERKS', 'CHESTER', 'CUMBERLAND', 'DAUPHIN', 'DELAWARE', 'JUNIATA', 'LANCASTER', 'LEBANON', 'MIFFLIN', 'PERRY', 'SCHUYLKILL', 'YORK']],
'VA':[['ALEXANDRIA CITY', 'ARLINGTON', 'CLARKE', 'CULPEPER', 'FAIRFAX CITY', 'FALLS CHURCH CITY', 'FAUQUIER', 'FREDERICK', 'FREDERICKSBURG CITY', 'GREENE', 'HARRISONBURG CITY', 'LOUDOUN', 'MADISON', 'MANASSAS CITY', 'ORANGE', 'PAGE', 'PRINCE WILLIAM', 'RAPPAHANNOCK', 'ROCKINGHAM', 'SHENANDOAH', 'SPOTSYLVANIA', 'STAFFORD', 'WARREN', 'WINCHESTER CITY'], ['ACCOMACK', 'CAROLINE', 'CHARLES CITY', 'CHESAPEAKE CITY', 'CHESTERFIELD', 'COLONIAL HEIGHTS CITY', 'ESSEX', 'FRANKLIN', 'GLOUCESTER', 'HANOVER', 'HARRISONBURG CITY', 'HENRICO', 'HOPEWELL CITY', 'ISLE OF WIGHT', 'JAMES CITY', 'KING AND QUEEN', 'KING GEORGE', 'KING WILLIAM', 'LANCASTER', 'MATHEWS', 'MIDDLESEX', 'NEW KENT', 'NEWPORT NEWS CITY', 'NORFOLK CITY', 'NORTHAMPTON', 'NORTHUMBERLAND', 'PETERSBURG CITY', 'POQUOSON CITY', 'PORTSMOUTH CITY', 'PRINCE GEORGE', 'RICHMOND', 'RICHMOND', 'SOUTHAMPTON', 'SUFFOLK CITY', 'SURRY', 'SUSSEX', 'VIRGINIA BEACH CITY', 'WESTMORELAND', 'WILLIAMSBURG CITY', 'YORK'], ['AMELIA', 'APPOMATTOX', 'BUCKINGHAM', 'CAMPBELL', 'CHARLOTTE', 'CUMBERLAND', 'DINWIDDIE', 'FLUVANNA', 'GOOCHLAND', 'LOUISA', 'NOTTOWAY', 'POWHATAN', 'PRINCE EDWARD'], ['BRUNSWICK', 'DANVILLE CITY', 'EMPORIA CITY', 'GREENSVILLE', 'HALIFAX', 'LUNENBURG', 'MECKLENBURG', 'PITTSYLVANIA'], ['ALBEMARLE', 'ALLEGHANY', 'AMHERST', 'AUGUSTA', 'BATH', 'BEDFORD', 'BOTETOURT', 'BUENA VISTA CITY', 'CHARLOTTESVILLE CITY', 'COVINGTON CITY', 'CRAIG', 'HIGHLAND', 'LEXINGTON CITY', 'LYNCHBURG CITY', 'NELSON', 'ROCKBRIDGE', 'STAUNTON CITY', 'WAYNESBORO CITY'], ['BLAND', 'BRISTOL', 'BUCHANAN', 'CARROLL', 'DICKENSON', 'FLOYD', 'FRANKLIN', 'GALAX CITY', 'GILES', 'GRAYSON', 'HENRY', 'LEE', 'MARTINSVILLE CITY', 'MONTGOMERY', 'NORTON CITY', 'PATRICK', 'PULASKI', 'RADFORD', 'ROANOKE', 'RUSSELL', 'SALEM', 'SCOTT', 'SMYTH', 'TAZEWELL', 'WASHINGTON', 'WISE', 'WYTHE']],
'WV':[['BOONE', 'CABELL', 'LINCOLN', 'LOGAN', 'MCDOWELL', 'MASON', 'MINGO', 'PUTNAM', 'RALEIGH', 'WAYNE', 'WYOMING'], ['BRAXTON', 'CALHOUN', 'CLAY', 'DODDRIDGE', 'GILMER', 'JACKSON', 'KANAWHA', 'LEWIS', 'PLEASANTS', 'RITCHIE', 'ROANE', 'TYLER', 'UPSHUR', 'WEBSTER', 'WIRT', 'WOOD'], ['BARBOUR', 'BROOKE', 'HANCOCK', 'HARRISON', 'MARION', 'MARSHALL', 'MONONGALIA', 'OHIO', 'PRESTON', 'RANDOLPH', 'TAYLOR', 'TUCKER', 'WETZEL'], ['BERKELEY', 'GRANT', 'HAMPSHIRE', 'HARDY', 'JEFFERSON', 'MINERAL', 'MORGAN', 'PENDLETON'], ['FAYETTE', 'GREENBRIER', 'MERCER', 'MONROE', 'NICHOLAS', 'POCAHONTAS', 'SUMMERS']]}
block_list = block_list_dict[state]
size = len(block_list)
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
I_df = I_df[I_df['State']==state]
#I_df = I_df.groupby(['COUNTY', 'YYYY'])['TotalDrugReportsCounty'].mean()
if drug == 'Heroin':
    I_df = I_df[I_df['SubstanceName']=='Heroin']
elif drug == 'SyntheticOpioid':
    I_df = I_df[I_df['SubstanceName']!='Heroin']
else:
    I_df = I_df[I_df['SubstanceName']==drug]
I_df = I_df.groupby(['COUNTY', 'YYYY'])['DrugReports'].sum()
def I_of_block(i, t):
    county_list = block_list[i]
    result = 0
    for county in county_list:
        if (county, time_map_dict[t]) in I_df.index:
            result += I_df[county, time_map_dict[t]]
    return result
I_dict = {}
for i in range(0, size):
    for t in range(0, max_time):
        I_dict[(i, t)] = I_of_block(i, t)

population_df = pd.read_csv('ACS_10_5YR_DP02_with_ann.csv')
population_df = population_df.iloc[1:]
population_df['HC01_VC128'] = population_df['HC01_VC128'].apply(lambda x:int(x))
population_df['State'] = population_df['GEO.display-label'].apply(lambda x:full2abbrev_dict[x.split(', ')[1]])
population_df = population_df[population_df['State']==state]
population_df['COUNTY'] = population_df['GEO.display-label'].apply(lambda x:x.split(', ')[0].upper().replace(' COUNTY', ''))
population_df = population_df.set_index('COUNTY')
population_df = population_df[['HC01_VC128']]
def population_of_block(i):
    county_list = block_list[i]
    result = 0
    for county in county_list:
        result += int(population_df.loc[county])
    return result
population_list = []
for i in range(0, size):
    population_list.append(population_of_block(i))



initial_state = []
for i in range(0, size):
    initial_state.append(I_of_block(i, 0))
'''
gamma = np.random.rand(size)
beta = np.random.rand()
A = np.random.rand(size, size)
'''
arg_sizes = [size*size, size]
total_size = sum(arg_sizes)
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
            if i == 1:#强行改
                bounds.append((0.3, 1))
            else:
                bounds.append((0.3, 1))
        else:
            bounds.append((0, 1))
            
bias += arg_sizes[0]
for i in range(bias, bias+arg_sizes[1]):
    bounds.append((0, 0.7))
bias += arg_sizes[1]

def get_gamma(args):
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
        result = (-summed_step(i, t, step-1, args)*S_step(i, t, step-1, args) + get_gamma(args)[i]*I_step(i, t, step-1, args))/steps_per_t + S_step(i, t, step-1, args)

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
        result = initial_state[i]
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
        '''
        ratio = max(1-get_gamma(args)[i]*I(i, t-1, args)/I(i, max(t-2, 0), args), 0)
        result = -summed(i, t-1, args)*S(i, t-1, args)*ratio + get_gamma(args)[i]*I(i, t-1, args) + S(i, t-1, args)
        '''
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
fastI_bar = lambda i, t:I_dict[(i, t)]
def dict_clear():
    I_results.clear()
    S_results.clear()
    summed_results.clear()
def f(args):
    result = 0
    for i in range(0, size):
        for t in range(0, train_time):
            if fastI_bar(i, t) != 0:
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
def plot(args):
    time_list_append = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
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
        pyplot.title('District {0} of {1}'.format(i+1, state))
        pyplot.legend(['predict', 'real'])
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
    args = np.random.rand(total_size)
    print(f(args))
    print(time.time()-start)
'''


result = scipy.optimize.differential_evolution(f, bounds, recombination=1, disp=True, maxiter=50)
pickle.dump(result, open('result(county block)', 'wb'))

#result = pickle.load(open('result(county block)', 'rb'))

plot(result.x)
x = result.x.copy()
for i,_ in enumerate(get_gamma(x)):
    get_gamma(x)[i] = 1
plot(x)

print('train error:'+str(f(result.x)))
#print('test error:'+str(f_test(result.x)))
print(get_arr_A(result.x))
print(get_gamma(result.x))

print('!')

print(get_arr_A(x))
print(get_gamma(x))