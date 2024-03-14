# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:54:49 2019

@author: vincentkuo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch

triger = 4
if triger==1:
    TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191028\TrainDataOrderByIV_Y191028_new_STD2.csv",encoding='ISO-8859-15')
    train_x = TrainSet.iloc[:,16:19].values
    train_y = TrainSet.iloc[:,-1].values
#TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191015\RFM_RecodeOutlier_Standize.csv",encoding='ISO-8859-15')
if triger==3:
    TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191101\TakeALookData2_Cluster_STD.csv",encoding='ISO-8859-15')
    train_x = TrainSet.iloc[:,116:119].values
    train_y = TrainSet.iloc[:,4].values
if triger==4:
    TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191101\TakeALookData3.csv",encoding='utf-16')
    train_x = TrainSet.iloc[:,6:8].values
    train_y = TrainSet.iloc[:,3].values

head = TrainSet.columns.values

'''
train_x = TrainSet.iloc[:,3:6].values
train_y = TrainSet.iloc[:,2].values
train_y[train_y=='Y']=1
train_y[train_y=='N']=0


plt.scatter(train_x[:, 0], train_x[:, 1], marker='o', c=train_y)
plt.show()
'''

y_pred = Birch(n_clusters = 6).fit_predict(train_x)
#y_pred = Birch(n_clusters = None, threshold=1.2, branching_factor=300).fit_predict(train_x)
y_distinct = len(set(y_pred))
print(y_distinct,"長度")
plt.scatter(train_x[:, 0], train_x[:, 1], marker='o', c=y_pred)
plt.show()
'''
plt.scatter(train_x[:, 0], train_x[:, 2], marker='o', c=y_pred)
plt.show()
plt.scatter(train_x[:, 1], train_x[:, 2], marker='o', c=y_pred)
plt.show()
'''

from sklearn import metrics
print("CH指標:", metrics.calinski_harabaz_score(train_x, y_pred))


#from sklearn import cluster, metrics
#dataSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191028\AllDataOrderByIV_Y191028_new_STD4Elbow.csv",encoding='ISO-8859-15')
#data = dataSet.iloc[:,11:76]
#for i in range (1,7):
 #   clusters = dataSet.iloc[:,i+3]
 #   silhouette_avg=metrics.silhouette_score(data,clusters)
 #   print(silhouette_avg)
