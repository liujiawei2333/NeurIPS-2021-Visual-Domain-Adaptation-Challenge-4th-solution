from os import name
import pandas as pd

path = 'final_file'
name = 'adapt_pred'

df = pd.read_excel('./%s/xlsx/%s.xlsx' % (path,name),  header=None)
df.to_csv('./%s/txt/%s.txt' % (path,name), header=None, sep=' ', index=False)