# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:59:28 2019

@author: vincentkuo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering

TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191015\TrainDataOrderByIV_Y191015_new_STD.csv",encoding='ISO-8859-15')
#TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191015\RFM_RecodeOutlier_Standize.csv",encoding='ISO-8859-15')

head = TrainSet.columns.values

train_x = TrainSet.iloc[:,5:70].values
train_y = TrainSet.iloc[:,4].values

'''
train_x = TrainSet.iloc[:,3:6].values
train_y = TrainSet.iloc[:,2].values
train_y[train_y=='Y']=1
train_y[train_y=='N']=0
'''

plt.scatter(train_x[:, 0], train_x[:, 1], marker='o', c=train_y)
plt.show()

y_pred = AgglomerativeClustering(n_clusters = None, distance_threshold=250, linkage='ward',).fit(train_x)
'''
y_distinct = len(set(y_pred))
print(y_distinct,"長度")
plt.scatter(train_x[:, 10], train_x[:, 11], marker='o', c=y_pred)
plt.show()

from sklearn import metrics
print("CH指標:", metrics.calinski_harabaz_score(train_x, y_pred))
'''
cluster = y_pred.labels_
print(cluster)
plt.scatter(train_x[:, 10], train_x[:, 11], marker='o', c=y_pred.labels_)
plt.show()

print(len(set(cluster)))