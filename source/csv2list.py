import pandas as pd

state = 'VA'
block_count = 6

df = pd.read_csv('{0}_blocks.csv'.format(state), header=None)
df.iloc[:,0] = df.iloc[:,0].map(lambda s:s.split(' County, ')[0].upper())

block_list = []
for i in range(0, block_count):
    county_list = []
    d = df[df.iloc[:, 1]==i+1]
    for j in range(0, d.shape[0]):
        county_list.append(d.iloc[j,0])
    block_list.append(county_list)
print(block_list)
open('{0}_blocks.txt'.format(state),'w').write(str(block_list))