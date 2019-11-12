from matplotlib import pyplot
import numpy as np 
import pandas as pd
import math

df = pd.read_csv('MCM_NFLIS_Data.csv')
df = df.groupby(['COUNTY', 'State', 'YYYY'])['TotalDrugReportsCounty'].mean()
state_list = ['KY','OH','PA','VA','WV']
for state in state_list:
    state_df = df[df.index.map(lambda x:x[1])==state]
    county_set = set()
    for index in state_df.index:
        county_set.add(index[0])
    
    for county in list(county_set):
        county_df = state_df[state_df.index.map(lambda x:x[0])==county]
        pyplot.plot(county_df.index.map(lambda x:x[2]).values, county_df.values)
        
    pyplot.yscale('log')
    pyplot.ylabel('TotalDrugReportsCounty')
    pyplot.xlabel('Year')
    pyplot.title('State: {0}'.format(state))
    pyplot.show()
print(df)
