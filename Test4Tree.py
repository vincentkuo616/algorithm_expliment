# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:18:24 2019

@author: vincentkuo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from xgboost import XGBClassifier
import pandas as pd

def confusionMatrix(yy, answer):
    
    C_matrix = confusion_matrix(yy,answer) ## 混淆矩陣 (yy:正確值;test2:預測值)
    print(C_matrix) ## show 混淆矩陣
    acc = round(100*(C_matrix[0,0]+C_matrix[1,1])/(C_matrix[0,0]+C_matrix[1,1]+C_matrix[0,1]+C_matrix[1,0]),5)
    print('ACC:', acc, '%')
    ppv = round(100*C_matrix[1,1]/(C_matrix[1,1]+C_matrix[1,0]),5)
    print('Precision:', ppv, '%')
    recall = round(100*C_matrix[1,1]/(C_matrix[1,1]+C_matrix[0,1]),5)
    print('Recall:', recall, '%')
    fScore = round(2/((1/ppv)+(1/recall)),5)
    print('F-Score:', fScore, '%')
    return C_matrix

TestSet  = pd.read_csv(r"C:\Users\vincentkuo\Documents\Test_Y190919.csv",encoding='utf-8')
X_2 = TestSet.loc[:,TestSet.dtypes!='object']
X_2 = X_2.iloc[:,3:].values
y_answer = TestSet.iloc[:,2].values
y_end = [np.zeros(10164)]

for i in range(0,49):
    TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\RUS_DATA\NEW_RUS_DATA_"+str(i)+".csv",encoding='utf-8')
    X = TrainSet.iloc[:,1:].values
    y = TrainSet.iloc[:,0].values
    regr_2 = tree.DecisionTreeClassifier()
    #regr_2 = RandomForestClassifier()
    regr_2 = XGBClassifier()
    regr_2.fit(X, y)

    # Predict
    y_2 = regr_2.predict(X_2)
    #print(y_2)
    #print("***********************************")
    y_end = np.append(y_end,[y_2],axis=0)
    #print(y_end)
#print(y_end)
avg = y_end[1,:]
#print(avg)
for i in range(2,len(y_end)):
    avg = avg + y_end[i,:]
print(avg)
avg = avg/49

for i in range(0,len(avg)):
    if(avg[i]>=0.5):
        avg[i]=1
    else:
        avg[i]=0
#print(avg)
confusionMatrix(avg,y_answer)

# Create the dataset
#rng = np.random.RandomState(1)

#y[y==1]='Y'
#y[y==0]='N'
#yy = [str(i) for i in y]

# Fit regression model
#regr_1 = DecisionTreeRegressor(max_depth=4)

#regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
 #                         n_estimators=300, random_state=rng)

# Plot the results
#plt.figure()
#plt.scatter(X, y, c="k", label="training samples")
#plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
#plt.xlabel("data")
#plt.ylabel("target")
#plt.title("Boosted Decision Tree Regression")
#plt.legend()
#plt.show()