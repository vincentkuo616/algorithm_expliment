# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:50:55 2019

@author: syn007
"""
#Glossary : https://docs.featuretools.com/usage_tips/glossary.html

import featuretools as ft
import pandas as pd
from featuretools.variable_types import Numeric
from featuretools.primitives import make_agg_primitive , make_trans_primitive
import time
tStart = time.time() #計時開始

#alldata = pd.read_csv('C:\\Users\\syn007\\Desktop\\test\\AllColumn4Test\\CreditScoring(1Kx5).csv')
#dfAllData_1 = alldata.iloc[:,100:] #Get column from the 1001 column
#dfAllData_1 = alldata
#print(dfAllData_1)   
 
# An EntitySet is a collection of entities and the relationships between them. 
# They are useful for preparing raw, structured datasets for feature engineering.
es = ft.demo.load_mock_customer(return_entityset=True)

print(es)

print(es["sessions"])

feature_matrix, feature_defs = ft.dfs(entityset=es,
                                       target_entity="customers",
                                       #agg_primitives=["mean", "sum", "mode"],
                                       #trans_primitives=["month", "hour"],
                                       max_depth=2)

#print(ft.primitives.list_primitives())

# == The ouput of DFS is a feature matrix and the corresponding list of feature definitions. ==

#print(feature_matrix[['MEAN(esAllData2.SUM(esAllData.Age))']].head(5))
tEnd = time.time() #計時結束
print(feature_defs)
print(feature_matrix[['MEAN(sessions.SUM(transactions.amount))']])
print(feature_matrix)
print(tEnd-tStart)
#print(ft.primitives.list_primitives())

#feature_matrix.to_csv('C:\\Temp\\CreditScoring(1Kx5)_depth2.csv')

#Glossary : https://docs.featuretools.com/usage_tips/glossary.html
