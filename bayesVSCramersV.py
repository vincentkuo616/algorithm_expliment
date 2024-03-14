# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:32:39 2021

@author: vincentkuo
"""

import numpy as np
import pandas as pd
from itertools import permutations, combinations

AX = 0.3
BX = 0.5
AY = 0.8

def cramersV(array):
    try:
        # print(array)
        A = array[0][0]+array[0][1]
        B = array[1][0]+array[1][1]
        X = array[0][0]+array[1][0]
        Y = array[0][1]+array[1][1]
        TTL = A+B
        cramersV = np.sqrt((array[0][0]-(A*X/TTL))**2/(A*X)+(array[0][1]-(A*Y/TTL))**2/(A*Y)
                           +(array[1][0]-(B*X/TTL))**2/(B*X)+(array[1][1]-(B*Y/TTL))**2/(B*Y))
    except Exception as e:
        print(e)
        
    return cramersV

def getBY(AX,BX,AY):
    try:
        if AX>1 or BX>1 or AY>1:    return 'Error'
                
    except Exception as e:
        print(e)
        
    return (AY*(1-AX)/AX)/((AY*(1-AX)/AX)+((1-AY)*(1-BX)/BX))

def bayes(prop,propT):
    try:
        test2 = list(permutations([1,2],2))
        prop_list = []
        propT_list = []
        if(prop.shape==(2,2)):
            for j in test2:
                prop_sum=0
                prop_sum=(prop[0][int(j[0])-1]+prop[1][int(j[1])-1])/2
                
                propT_sum=(propT[0][int(j[0])-1]+propT[1][int(j[1])-1])/2
                    
                prop_list.append(prop_sum)
                propT_list.append(propT_sum)
        
    except Exception as e:
        print(e)
        
    return max(prop_list)/2+max(propT_list)/2

def getChiSquare(array):
    try:
        from scipy.stats import chi2_contingency
        
        chiSquare = chi2_contingency(array)[0]
        pValue    = chi2_contingency(array)[1]
                   
    except Exception as e:
        print(e)
        
    return chiSquare

def getMLChiSquare(array):
    try:
        A = array[0][0]
        B = array[1][0]
        C = array[0][1]
        D = array[1][1]
        A_E = (A+C)*(A+B)/(A+B+C+D)
        B_E = (A+B)*(B+D)/(A+B+C+D)
        C_E = (A+C)*(C+D)/(A+B+C+D)
        D_E = (B+D)*(C+D)/(A+B+C+D)
        print(A,B,C,D)
        print(A_E,B_E,C_E,D_E)
        
        mlChiSquare = 2*A*np.log(A/A_E)+2*B*np.log(B/B_E)+2*C*np.log(C/C_E)+2*D*np.log(D/D_E)

    except Exception as e:
        print(e)
        
    return mlChiSquare

array = np.array([[200*AY,200*(1-AY)],[200*AY*(1-AX)/AX,200*(1-AY)*(1-BX)/BX]])

prop = array / array.sum(axis=1)[:, np.newaxis]
propT = array.transpose() / array.sum(axis=0)[:, np.newaxis]


print(cramersV(array))
print(bayes(prop,propT))
outputArray = []

for i in range (99):
    AX = round(i/100+0.01,2)
    for j in range(99):
        BX = round(j/100+0.01,2)
        for k in range(99):
            AY = round(k/100+0.01,2)
            array = np.array([[200*AY,200*(1-AY)],[200*AY*(1-AX)/AX,200*(1-AY)*(1-BX)/BX]])
            prop = array / array.sum(axis=1)[:, np.newaxis]
            propT = array.transpose() / array.sum(axis=0)[:, np.newaxis]
            listProp = [AX,BX,AY,round(getBY(AX,BX,AY),2)]
            listProp.sort()
            strlistProp = ''.join(str(e) for e in listProp)
            outputArray.append([AX,BX,AY,round(getBY(AX,BX,AY),2),cramersV(array),bayes(prop,propT),strlistProp])
            # print('AX,',AX,',BX,',BX,',AY,',AY,',BY,',getBY(AX,BX,AY),',CV,',cramersV(array),',Bayes,',bayes(prop,propT))

column = ['AX占比','BX占比','AY占比','BY占比','CV','Bayes','PK']
outputData = pd.DataFrame(np.array(outputArray),columns=column)
#outputData.to_csv("C:\\Users\\vincentkuo\\Desktop\\AI團隊培訓\\MIS資料分析\\公司員工習性分析\\TEMP.csv",encoding='utf-16')


