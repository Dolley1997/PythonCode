# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:02:13 2020

@author: Dolley
"""

import glob
import gzip
import pandas as pd
from datetime import datetime
import os
from scipy import stats
import numpy as np
DIR = r'E:\DataSetsML\Machine Learning A-Z Template Folder\box'
#print([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])       
csvFiles = glob.glob("*.log")
print(csvFiles)
outfile='tag.csv'
val='2020-02-24 00:36:46.281'
to_dtime_obj = datetime.strptime('2020-01-17 15:36:46', '%Y-%m-%d %H:%M:%S')
from_dtime_obj = datetime.strptime('2020-01-16 15:36:46', '%Y-%m-%d %H:%M:%S')
for files in csvFiles:
    csv = open(outfile, "w")
    csv.write('Time,Tag,Reader,Antenna,rssi\n')
    with open(files,'rt') as f:
        for line in f:
            if('tagId : ' in line and 'rdr : ' in line and 'antenna : ' in line):
                #str_Chck = line.split(",")
                #val=str_Chck[0]
                #check_date= datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                line = line.replace(',','.')
                #match = re.match(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}).(\d{3})', line)
                #date=str(match)
                #if(check_date >= from_dtime_obj and check_date <= to_dtime_obj):
                line=line.replace(' INFO taglogger [AsyncMessageHandler] tagId : ',',')
                line=line.replace('  rdr : ',',')
                line=line.replace('  antenna : ', ',')
                line=line.replace('  rssi : ',',')
                csv.write(line)
    csv.close()
    print("csv writing is done")
    
"""    
dataset = pd.read_csv('Tag.csv')
dataset.describe()
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
z = np.abs(stats.zscore(dataset))
print(z)
print(dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values    
         
"""