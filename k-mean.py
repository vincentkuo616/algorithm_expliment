# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:27:40 2020

@author: syn006
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans                       



# mse_list格式要為  mse_list = ( [分群數 , MSE] )
def elbow (mse_list):
    # 找出linear y=ax+b
    leng=len(mse_list)
    X0 = float(mse_list[0][1])
    Y0 = 0
    X_max = float(mse_list[leng-1][1])
    Y_max = leng-1

    a =  (X_max-X0)/(Y_max-Y0)
    b = X0 
    #找出線上的每一點
    pred_mse =[]
    for i in range(0,leng,1):
        pred = a * i +b
        pred_mse.append(pred)
    #計算差異
    diff=[]   

    for i in range(0,leng,1):
        a=pred_mse[i]
        b=mse_list[i][1]
        diff_tmp = a-b
        arr=[i,diff_tmp]
        diff.append(arr)
    """
    diff = ([0,0],
    [1,-0.888888888888889],
    [2,-0.777777777777779],
    [3,-0.666666666666666],
    [4,-1.55555555555556],
    [5,-2.44444444444444],
    [6,1.66666666666667],
    [7,0.777777777777779],
    [8,0.888888888888889],
    [9,0])
    """
    #找出最佳轉折點
    index, value =max(diff, key=lambda item: item[1])
    
    a = index
    if( a<1 ) : 
        a=1
    return a


TW = pd.read_csv("C:\\Users\\vincentkuo\\Documents\\temp4Kmeans\\TW.csv",encoding='utf-16')
HK = pd.read_csv("C:\\Users\\vincentkuo\\Documents\\temp4Kmeans\\HK.csv",encoding='utf-16')
AU = pd.read_csv("C:\\Users\\vincentkuo\\Documents\\temp4Kmeans\\AU.csv",encoding='utf-16')

ll = ['1','2','3']

df_all = [TW,HK,AU]

for area in ll:
    aa = 'AU'
    df=AU
    if (area=='1'):
        df=TW
        aa='TW'
    if (area=='2'):
        df=HK  
        aa='HK'
 
    #取得所有PSCODE
    uni_ps = np.unique( df['PS_CODE'])
    col_name=df.columns.tolist()
    #建立輸出結果
    result = pd.DataFrame(columns=col_name)
    #對每個PSCODE分群
    for ps_code in uni_ps:
    
        dataGroup=df.groupby('PS_CODE')
        dataSelected = dataGroup.get_group(ps_code)
        col=['cost_percent']
        X=np.array( dataSelected[col] )
        #資料過少就跳過
        if(len(dataSelected)<5):
            continue  
        #找出最佳分群
        k_list=[]
        sse_list = [ ] 
        max_group = len(X)
        maxK=15
        if(max_group<15):
            maxK=max_group
        
        for k in range(1,maxK): 
            kmeans=KMeans(n_clusters=k, n_jobs = 6) 
            kmeans.fit(X) 
            tmp_list = [k,kmeans.inertia_]#model.inertia_返回模型的误差平方和，保存进入列表
            sse_list.append(tmp_list)   
            k_list.append(kmeans.labels_)#把結果存在這邊
        bestK = elbow(sse_list)
        binned = k_list[bestK] #最佳分群結果
        dataSelected['kmeans']=0
        dataSelected['kmeans'] = binned
        result = result.append(dataSelected)
    #落地
    result.to_csv("C:\\Users\\vincentkuo\\Documents\\temp4Kmeans\\"+aa+"_result.csv",sep=",",index=False)
