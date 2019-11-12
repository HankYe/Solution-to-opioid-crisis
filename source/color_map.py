### color_map.py
 
from bs4 import BeautifulSoup
import pandas as pd
 
FIPS_State_list = ['51', '39', '42', '21', '54']

drug = 'All'
reports = {}

I_df = pd.read_csv('MCM_NFLIS_Data.csv')
#I_df = I_df[I_df['State']==state]
#I_df = I_df.groupby(['COUNTY', 'YYYY'])['TotalDrugReportsCounty'].mean()
if drug == 'All':
    pass
elif drug == 'SyntheticOpioid':
    I_df = I_df[I_df['SubstanceName']!='Heroin']
else:
    I_df = I_df[I_df['SubstanceName']==drug]
I_df = I_df.groupby(['FIPS_Combined'])['DrugReports'].sum()
for fips in I_df.index:
    reports[str(fips)] = I_df[fips]


 
 
# Load the SVG map
svg = open('counties.svg', 'r').read()
 
# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])
 
# Find counties
paths = soup.findAll('path')
 
# Map colors
colors = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]
 
# County style
path_style = '''font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;
stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;
marker-start:none;stroke-linejoin:bevel;fill:'''
 
# Color the counties based on unemployment rate
for p in paths:
    FIPS = p['id'].strip('FIPS_')
    if p['id'] not in ["State_Lines", "separator"]:
        if FIPS[0:2] not in FIPS_State_list:
            continue
        try:
            rate = reports[FIPS]
        except:
            rate = 0
             
         
        if rate > 5000:
            color_class = 5
        elif rate > 1000:
            color_class = 4
        elif rate > 200:
            color_class = 3
        elif rate > 40:
            color_class = 2
        elif rate > 8:
            color_class = 1
        else:
            color_class = 0
 
 
        color = colors[color_class]
        p['style'] = path_style + color

open('out.svg', 'w').write(soup.prettify()) 

