# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:15:58 2023

@author: vincentkuo
"""

# need to import these things first 
from sklearn.datasets import load_iris
import numpy as np 
import pandas as pd
# use load_iris 
iris = load_iris()
# convert into a pandas data frame
df = pd.DataFrame(
  data= np.c_[iris['data'], iris['target']], 
  columns= iris['feature_names'] + ['species']
)
# set manually the species column as a categorical variable
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
# use ".head()" to show the first 5 rows
print(df.head())

# load the ydata_profiling package
from ydata_profiling import ProfileReport
# use ProfileReport
pr_df = ProfileReport(df)
# show pr_df
#print(pr_df)

'''
ydata-profiling   dtale   autoviz   sweetviz
'''