from numpy.core.fromnumeric import argmax
import pandas as pd
import os
import numpy as np
from pandas.core.frame import DataFrame

path = 'final_file'
save_name = 'adapt_pred'

dir = os.chdir('./%s/xlsx/' % path)

file_test = os.listdir(dir)

file_new0 = pd.DataFrame()
file_new1 = pd.DataFrame()
file_new2 = pd.DataFrame()

for fil in file_test:
    file_fir = pd.read_excel(fil,header=None)
    file_fir.columns = ['mulu','type','score']
    file0 = file_fir.loc[:,['mulu']]
    file1 = file_fir.loc[:,['type']]
    file2 = file_fir.loc[:,['score']]
    file_new0 = pd.concat([file0,file_new0],axis=1)
    file_new1 = pd.concat([file1,file_new1],axis=1)
    file_new2 = pd.concat([file2,file_new2],axis=1)

column1 = []
column2 = []
for i in range(0,len(file_test)):
    se1 = 'type' + str(i+1)
    se2 = 'score' + str(i+1)
    column1.append(se1)
    column2.append(se2)

file_new1.columns = column1
file_new2.columns = column2

se1 = []
se2 = []
se3 = []

for row in range(0,len(file_new1)):
    list1 = file_new1.iloc[row,:].tolist()
    l1 = len(list1)
    cnt = np.bincount(list1)
    if max(cnt) == 1:
        a = list1[0]
    else:
        a = argmax(np.bincount(list1))

    se1.append(a)
    id1 = [i for i,x in enumerate(list1) if x == a]
    if 0 in id1:
        b = 0
    else:
        b = id1[0]
    se2.append(b)


for i in range(0,len(file_new2)):
    poi = se2[i]
    poi_column = 'score' + str(poi + 1)
    best_val = file_new2.at[i,poi_column]
    se3.append(best_val)

dex = file_new0.iloc[:,0].tolist()
df = {
    'dex':dex,
    'max':se1,
    'poi':se2,
    'val':se3
}

file_new3 = DataFrame(df)
file_all = file_new3[['dex','max','val']]
file_all.to_excel('%s.xlsx' % save_name)









