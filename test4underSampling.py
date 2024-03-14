# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:53:28 2019

@author: vincentkuo
"""

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler,NearMiss,EditedNearestNeighbours,RepeatedEditedNearestNeighbours,AllKNN,TomekLinks,CondensedNearestNeighbour,NeighbourhoodCleaningRule
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.ensemble import EasyEnsemble
from collections import Counter
#from sklearn.datasets import make_classification


TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191028\TrainData_B_Subset3.csv",encoding='utf-8')
#TestSet  = pd.read_csv(r"C:\Users\vincentkuo\Documents\Test_Y190919_2.csv",encoding='utf-8')

head = TrainSet.columns.values

X = TrainSet.iloc[:,6:].values
y = TrainSet.iloc[:,5].values
#X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
 #                          n_redundant=0, n_repeated=0, n_classes=3,
  #                         n_clusters_per_class=1,
   #                        weights=[0.1, 0.3, 0.6],
    #                       class_sep=0.8, random_state=0)
#print(X)
#print(y)
print(Counter(y))

#下采样

#原型生成算法
#给定数据集S, 原型生成算法将生成一个子集S1,其中|S1| < |S|, 但是子集并非来自于原始数据集.意思就是说: 原型生成方法将减少数据集的样本数量,剩下的样本是由原始数据集生成的,而不是直接来源于原始数据集.
#ClusterCentroids函数实现了上述功能: 每一个类别的样本都会用K-Means算法的中心点来进行合成, 而不是随机从原始样本进行抽取.

#cc=ClusterCentroids(random_state=0)
#X_centroids,y_centroids=cc.fit_sample(X,y)
#print(Counter(y_centroids))

#原型选择算法
#原型选择算法是直接从原始数据集中进行抽取
#RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据:随机选取数据的子集.
#通过设置RandomUnderSampler中的replacement=True参数, 可以实现自助法(Bootstrap)有放回抽样.
######################################################
#rus=RandomUnderSampler(random_state=0,replacement=True)
#X_rus,y_rus=rus.fit_sample(X,y)
######################################################
import time
s = time.time()
allknn = AllKNN(random_state=0)
X_ru,y_ru=allknn.fit_sample(X,y)
ada = SMOTE(random_state=0, ratio=0.09)
X_rus,y_rus=ada.fit_sample(X_ru,y_ru)
array = np.column_stack([y_rus,X_rus])
print(Counter(y_ru))
print(Counter(y_rus))
data2 = pd.DataFrame(array, columns=np.append(head[5],head[6:]))

data2.to_csv("C:\\Users\\vincentkuo\\Documents\\vincent\\Y191028\\TrainData_AllKNNSMOTE_Y191030_new.csv",index=False)
'''
for i in range(1,2):
    allknn = TomekLinks(random_state=i)
    X_rus,y_rus=allknn.fit_sample(X,y)
    mergeData = np.column_stack([y_rus,X_rus])
    print(Counter(y_rus))
    print(array)
    print(mergeData)
    #array = np.append(array,mergeData,axis=0)
    data = pd.DataFrame(mergeData, columns=head[2:])
    #data.to_csv("C:\\Users\\vincentkuo\\Documents\\TomekLinks_"+str(i)+".csv",index=False)
    print(data.shape)
    if(i==48):
        x=1
        #data.to_csv("C:\\Users\\vincentkuo\\Documents\\RUS_Data_"+str(i)+".csv",index=False)
'''
#merge = pd.merge(TrainSet,data,on=2)
#print(merge)

#data.to_csv("C:\\Users\\vincentkuo\\Documents\\mergeData.csv")

#NearMiss函数则添加了一些启发式(heuristic)的规则来选择样本, 通过设定version参数来实现三种启发式的规则.
#NearMiss-1: 选择离N个近邻的负样本的平均距离最小的正样本;
#NearMiss-2: 选择离N个负样本最远的平均距离最小的正样本;
#NearMiss-3: 是一个两段式的算法. 首先, 对于每一个负样本, 保留它们的M个近邻样本; 接着, 那些到N个近邻样本平均距离最大的正样本将被选择.

#nm1 = NearMiss(random_state=0, version=1)
#X_nm1, y_nm1 = nm1.fit_sample(X, y)
#print(Counter(y_rus))

#Cleaning under-sampling techniques
#TomekLinks : 样本x与样本y来自于不同的类别,满足以下条件,它们之间被称之为TomekLinks;不存在另外一个样本z,使得d(x,z)<d(x,y)或者 d(y,z)< d(x,y)成立.其中d(.)表示两个样本之间的距离,也就是说两个样本之间互为近邻关系.
#这个时候, 样本x或样本y很有可能是噪声数据, 或者两个样本在边界的位置附近.
#TomekLinks函数中的auto参数控制Tomek's links中的哪些样本被剔除. 默认的ratio='auto' 移除多数类的样本, 当ratio='all'时, 两个样本均被移除.
#EditedNearestNeighbours这种方法应用最近邻算法来编辑(edit)数据集, 找出那些与邻居不太友好的样本然后移除. 对于每一个要进行下采样的样本, 那些不满足一些准则的样本将会被移除;
#他们的绝大多数(kind_sel='mode')或者全部(kind_sel='all')的近邻样本都属于同一个类, 这些样本会被保留在数据集中.
enn = EditedNearestNeighbours(random_state=0)
renn = RepeatedEditedNearestNeighbours(random_state=0)
allknn = AllKNN(random_state=0)


#smote_tomek = SMOTETomek(random_state=0)
#x_tomek, y_tomek = smote_tomek.fit_sample(X, y)
#print(Counter(y_tomek))

#Ensemble例子

#EasyEnsemble 有两个很重要的参数:(i)n_subsets 控制的是子集的个数(ii)replacement决定是有放回还是无放回的随机采样.
#ee = EasyEnsemble(random_state=0,n_subsets=10,replacement=True)
#X_ee, y_ee = ee.fit_sample(X, y)
#print(X_ee.shape)
#print(Counter(y_ee[0]))
e = time.time()
print(e-s)