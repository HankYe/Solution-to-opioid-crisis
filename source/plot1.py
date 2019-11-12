from matplotlib import pyplot
import numpy as np 
import pandas as pd
import math

df = pd.read_csv('MCM_NFLIS_Data.csv')
df = df.groupby(['State', 'YYYY'])['TotalDrugReportsState'].mean()
state_list = ['KY','OH','PA','VA','WV']
for state in state_list:
    state_df = df[df.index.map(lambda x:x[0])==state]
    pyplot.plot(state_df.index.map(lambda x:x[1]).values, state_df.values)
        
        
#pyplot.yscale('log')
pyplot.ylabel('TotalDrugReportsState')
pyplot.xlabel('Year')
pyplot.title('States')
pyplot.legend(state_list)
pyplot.show()
print(df)