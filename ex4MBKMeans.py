# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:43:22 2019

@author: vincentkuo
"""
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
from collections import Counter

def checkFrequency(array):
    a_Y,a_N,b_Y,b_N,c_Y,c_N,d_Y,d_N=0,0,0,0,0,0,0,0
    for i in range (0,len(array)):
        if array[i][-2]==0 and array[i][-1]==0:
            a_N+=1
        elif array[i][-2]==0 and array[i][-1]==1:
            a_Y+=1
        elif array[i][-2]==1 and array[i][-1]==0:
            b_N+=1
        elif array[i][-2]==1 and array[i][-1]==1:
            b_Y+=1
        elif array[i][-2]==2 and array[i][-1]==0:
            c_N+=1
        elif array[i][-2]==2 and array[i][-1]==1:
            c_Y+=1
        #elif array[i][3]==3 and array[i][4]==0:
         #   d_N+=1
        #elif array[i][3]==3 and array[i][4]==1:
         #   d_Y+=1
        else:
            print('What?!?! Something Wrong!!')
    print('a_Y:',a_Y,'  a_N',a_N,'  a_Y%',a_Y/(a_Y+a_N),'  a_N%',a_N/(a_Y+a_N))
    print('b_Y:',b_Y,'  b_N',b_N,'  b_Y%',b_Y/(b_Y+b_N),'  b_N%',b_N/(b_Y+b_N))
    print('c_Y:',c_Y,'  c_N',c_N,'  c_Y%',c_Y/(c_Y+c_N),'  c_N%',c_N/(c_Y+c_N))
    #print('d_Y:',d_Y,'  d_N',d_N,'  d_Y%',d_Y/(d_Y+d_N),'  d_N%',d_N/(d_Y+d_N))


TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191015\TrainDataOrderByIV_Y191015_new_STD.csv",encoding='ISO-8859-15')
#TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191015\RFM_RecodeOutlier_Standize.csv",encoding='ISO-8859-15')
TestSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191015\TestDataOrderByIV_Y191015_new_STD.csv",encoding='ISO-8859-15')
AllDataSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191015\AllDataOrderByIV_Y191015_new_STD.csv",encoding='ISO-8859-15')

head = TrainSet.columns.values

train_x = TrainSet.iloc[:,20:23].values
train_y = TrainSet.iloc[:,4].values
test_x = TestSet.iloc[:,20:23].values
test_y = TestSet.iloc[:,4].values
All_x = AllDataSet.iloc[:,20:23].values
All_y = AllDataSet.iloc[:,4].values
triger=4
if triger==3:
    TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191101\TakeALookData2_Cluster_STD.csv",encoding='ISO-8859-15')
    train_x = TrainSet.iloc[:,116:119].values
    train_y = TrainSet.iloc[:,4].values
if triger==4:
    TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191101\TakeALookData3.csv",encoding='utf-16')
    train_x = TrainSet.iloc[:,6:8].values
    train_y = TrainSet.iloc[:,3].values

# fit on the whole data
kmeans = MiniBatchKMeans(n_clusters=6,
                         random_state=1,
                         batch_size=6,
                         max_iter=10).fit(train_x)
print(kmeans.cluster_centers_)
ans = kmeans.predict(train_x)
print(ans)
#print(Counter(ans))

ans_test = kmeans.predict(test_x)
print(ans_test)
'''
final = kmeans.predict(All_x)

array = np.column_stack([All_x,final])
array2 = np.column_stack([array,All_y])
'''
array = np.column_stack([train_x,ans])
array2 = np.column_stack([array,train_y])
'''
array = np.column_stack([test_x,ans_test])
array2 = np.column_stack([array,test_y])
'''
#print(array2)
#checkFrequency(array2)