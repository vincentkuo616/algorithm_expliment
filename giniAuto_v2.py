# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:02:01 2020

@author: syn009
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime


## 随机1000 个数据
np.random.seed(1) #种子只是为了重现结果

prjRootFolder = "C:\\vincent\\GINI&CV\\Y201229_Quota4GINI\\"
prjRootFolder = "C:\\Temp\\Dayz\\toU軒\\GINI_HHI\\"
perspective = ['OFFICE','OFFICE-CUST','OFFICE-CMR','OFFICE-CMR-CUST','CUST','CMR','CMR-CUST','CUSTTYPE-CUST','CUSTTYPE-CUST-PS','BRAND-PS','BRAND-PS-CUST','BRAND-CUST','OFFICE-BRAND','OFFICE-BRAND-CUST','PS-CUST']
perspective = ['BU-OFFICE','BU-OFFICE-CUST','BU-OFFICE-CMR','BU-OFFICE-CMR-CUST','BU-CUST','BU-CMR','BU-CMR-CUST','CUSTTYPE-CUST','CUSTTYPE-CUST-PS','BRAND-PS','BRAND-PS-CUST','BRAND-CUST','OFFICE-BRAND','OFFICE-BRAND-CUST','PS-CUST']

perspective = ['BRAND-CMR']
perspective = ['CS_GROUP-CMR','CS_GROUP-CUST']
perspective = ['CS_GROUP-CMR','BRAND-CMR','PS_CODE-CMR','CSG_BRAND-CMR','CS_GROUP-CUST','BRAND-CUST','PS_CODE-CUST','CSG_BRAND-CUST','OFFICE-CUST','OFFICE-CMR-CUST']
perspective = ['CUST-PS_CODE']
perspective = ['CS_GROUP-PS_CODE-CUST','CS_GROUP-PS_CODE-CMR']
perspective = ['CSG_BRAND-CUST','CSG_BRAND-CMR']
perspective = ['CSG-PRODPK-CUST-YYYYMM']
#perspective = ['BU-BU-CUST-YYYYMM']
perspective = ['CS_GROUP-PS_CODE-客戶代碼']
timePerspective = 'SEASON'
'''
    Save Data Informations
'''
outputName = ".xlsx"
saveFlag = 0

dataName = 'DBS_PMI4Python_less.csv'
dataName = 'DBS_PMI4Python.csv'
dataName = 'DBS_PMI4Python_WATST.csv'
dataName = 'DBS_PMI_CN_M.csv'
dataName = 'DBS_PMI_CN_L3M_v2.csv'
dataName = 'DBS_PMI_CN_L3M_v3.csv'
dataName = 'DBS_PMI_CN_L3M_v4.csv'
dataName = 'DBS_PMI_CN_L3M_v5.csv'
#dataName = 'TW_output4Gini\\DBS_PMI_TW_L3M.csv'
dataName = 'TW_output4Gini\\DBS_PMI_TW_L3M_v2.csv'
dataName = 'CMR_CUST4GINI.csv'
dataName = 'CUST_%4GINI_ZERO_REVERSE.csv'
dataName = 'CSG-PROD-CUST_%4GINI_REVERSE.csv'
dataName = 'CSG-PROD_4GINI.csv'
dataName = 'CUST_%4GINI_Outlier.csv'
dataName = 'CSG-PROD_4GINI_Outlier.csv'
dataName = 'CSG-PROD-CUST_%4GINI_Outlier.csv'
dataName = 'CSG-PS_AMT_Y20Q4_2.xlsx'
dataName = 'CSG-PS_AMT_Y20Q4.xlsx'

def main():
    dataRoot = prjRootFolder+dataName
    data = readData(dataRoot)
    #print(data.head())
    
    groupbyLayer(data,perspective)  #主要運算層
    
    '''
    dataGroup=data.groupby('4PythonGroup')
    for key in dataGroup.groups.keys():
        dataSelected = dataGroup.get_group(key)
        dataSelected = dataSelected['總銷售額'].iloc[:]
        print(key,"  -  ",GiniIndex(dataSelected))
        #print(GiniIndex(data))
    '''

'''
    Parameter:
        data        :RawData
        perspective :GroupBy條件
    Rule:
        YYMM  Layer1 Layer2 Amount
        2007     aaa    bb1  1000
        2007     aaa    bb1   100
        2007     aaa    bb2    10
        2007     aaa    bb2     5
        2007     aaa    bb1     5
        2007     aaa    bb1     3
        2007     aaa    bb3     2
                    *
                    *
                    V
        YYMM  Layer1 Layer2 Amount
        2007     aaa    bb1  1108
        2007     aaa    bb2    15
        2007     aaa    bb3     2    * * > Compute GINI
'''
def groupbyLayer(data,perspective):
    try:
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Group By Layer Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        for i in perspective:
            # 4 OutputData
            dataOutputLayer = []
            layers = len(i.split('-'))
            if layers == 1:
                layers1 = i
                if timePerspective=='YYMM': time4Name = 'M'
                else: time4Name = 'Q'
                columnName = '_AMT[BU_'+layers1+']_by'+time4Name
                dataOutputLayer.append([timePerspective,'BU','GINI'+columnName,'HHI'+columnName])
                
                dataSelectedLayer2 = data.groupby(layers1)['SALESAMOUNTNONVAT'].agg('sum')
                dataSelectedLayer2[dataSelectedLayer2<0] = 0
                gini=GiniIndex(dataSelectedLayer2)
                hhi =HHI(dataSelectedLayer2)
                if gini==0: gini=np.nan
                dataOutputLayer.append([timePerspective,'CN',gini,hhi])
            if layers == 2:
                layers1 = i.split('-')[0]
                layers2 = i.split('-')[1]
                if timePerspective=='YYMM': time4Name = 'M'
                else: time4Name = 'Q'
                columnName = '_AMT['+layers1+'_'+layers2+']_by'+time4Name
                dataOutputLayer.append([timePerspective,layers1,'GINI'+columnName,'HHI'+columnName])
                
                dataLayers1 = data.groupby(layers1)
                for index in dataLayers1.groups.keys():
                    dataSelectedLayer1 = dataLayers1.get_group(index)
                    dataSelectedLayer2 = dataSelectedLayer1.groupby(layers2)['SALESAMOUNTNONVAT'].agg('sum')
                    dataSelectedLayer2[dataSelectedLayer2<0] = 0
                    gini=GiniIndex(dataSelectedLayer2)
                    hhi =HHI(dataSelectedLayer2)
                    if len(dataSelectedLayer2)==1: gini=np.nan
                    dataOutputLayer.append([timePerspective,index,gini,hhi])
            if layers == 3:
                layers1 = i.split('-')[0]
                layers2 = i.split('-')[1]
                layers3 = i.split('-')[2]
                if timePerspective=='YYMM': time4Name = 'M'
                else: time4Name = 'Q'
                columnName = '_AMT['+layers1+'_'+layers2+'_'+layers3+']_by'+time4Name
                dataOutputLayer.append([timePerspective,layers1,layers2,'GINI'+columnName,'HHI'+columnName])
                
                dataLayers1 = data.groupby([layers1,layers2])
                for index in dataLayers1.groups.keys():
                    dataSelectedLayer1 = dataLayers1.get_group(index)
                    dataSelectedLayer2 = dataSelectedLayer1.groupby(layers3)['SALESAMOUNTNONVAT'].agg('sum')
                    dataSelectedLayer2[dataSelectedLayer2<0] = 0
                    gini=GiniIndex(dataSelectedLayer2)
                    hhi =HHI(dataSelectedLayer2)
                    if gini==0: gini=np.nan
                    dataOutputLayer.append([timePerspective,index[0],index[1],gini,hhi])
            if layers == 4:
                layers1 = i.split('-')[0]
                layers2 = i.split('-')[1]
                layers3 = i.split('-')[2]
                layers4 = i.split('-')[3]
                if timePerspective=='YYMM': time4Name = 'M'
                else: time4Name = 'Q'
                columnName = '_AMT['+layers1+'_'+layers2+'_'+layers3+'_'+layers4+']_by'+time4Name
                dataOutputLayer.append([timePerspective,layers1,layers2,layers3,'GINI'+columnName,'HHI'+columnName])
                
                dataLayers1 = data.groupby([layers1,layers2,layers3])
                for index in dataLayers1.groups.keys():
                    print(dataLayers1.groups.keys())
                    dataLayers1.get_group(index)
                    dataSelectedLayer1 = dataLayers1.get_group(index)
                    dataSelectedLayer2 = dataSelectedLayer1.groupby(layers4)['SALESAMOUNTNONVAT'].agg('sum')
                    dataSelectedLayer2[dataSelectedLayer2<0] = 0
                    '''
                    addZero = pd.Series([0])
                    while len(dataSelectedLayer2)<8:
                        dataSelectedLayer2=dataSelectedLayer2.append(addZero)
                    '''
                    gini=GiniIndex(dataSelectedLayer2)
                    hhi =HHI(dataSelectedLayer2)
                    if gini==0: gini=np.nan
                    dataOutputLayer.append([timePerspective,index[0],index[1],index[2],gini,hhi])
            if saveFlag==1:
                saveData(dataOutputLayer,i)
        print(dataOutputLayer)
    except:
        print('Exception in GroupbyLayer')
    
def saveData(data,dataName):
    try:
        timestr = getDatetimeStr()
        #outputNameFinal = outputName.split('.')[0]+dataName+'_'+timestr+'.'+outputName.split('.')[1]
        outputNameFinal = outputName.split('.')[0]+dataName+'.'+outputName.split('.')[1]
        outputData = pd.DataFrame(data[1:],columns=data[0])
        print(outputData)
        if outputName.split('.')[1]=='xlsx':
            location = prjRootFolder+outputNameFinal
            outputData.to_excel(location,index=False)
        elif outputName.split('.')[1]=='csv':
            outputData.to_csv(prjRootFolder+outputNameFinal,index=False)
        else:
            print(outputData)
            
        print('Save Succeed')
    except:
        print("SaveData failed")

def HHI(p):
    try:
        sum = np.sum(p)
        percent = p / sum
        hhi = np.sum(percent*percent)
    except:
        print("HHI Failed")
    return hhi

def GiniIndex(p):
    '''基尼系数'''
    #cum = np.cumsum(sorted(np.append(p, 0))) #遇到負的會有問題
    cum = np.cumsum(sorted(p))
    cum = np.append(0,cum)
    sum = cum[-1]
    x = np.array(range(len(cum))) / len(p)
    y = cum / sum
    B = np.trapz(y, x=x)
    A = 0.5 - B
    G = A / (A + B)
    #圖形修正
    #G = G + 1/len(p)
        
    '''绘图'''
    #'''
    from matplotlib import pyplot as plt
    from scipy.interpolate import make_interp_spline
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', -0))
    ax.spines['left'].set_position(('data', 0))
    plt.xticks([0, 1.0])
    plt.yticks([1.0])
    plt.axis('scaled')
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    ax.plot(x_smooth, y_smooth, color='black')
    ax.plot(x, x, color='black')
    ax.plot([0, 1, 1, 1], [0, 0, 0, 1], color='black')
    ax.fill_between(x, y)
    ax.fill_between(x, x, y, where=y <= x)

    plt.show()
    #'''    
    return G

def getDatetimeStr():
    d = datetime.now()
    str='{:%Y%m%d%H%M%S}'.format(d)
    return str

def readData(root):
    try:
        if root[-3:]=='csv':
            data = pd.read_csv(root,encoding='utf-16')
        elif root[-3:]=='lsx':
            data = pd.read_excel(root,encoding='utf-8')
        elif root[-3:]=='txt':
            data = pd.read_csv(root,header=None,encoding='utf-8')
        else:
            data = pd.read_excel(root,encoding='utf-8')
    except:
        print("Data parsing failed")
    return data

if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    a = [60,20,10,4,1,100,1,10,1,1]
    b = [91,1,1,1,1,1,1,1,1,1]
    c = [80,6,3,5,1,1,1,1,1,1]
    d = [19,17,15,13,11,9,5,7,3,1]
    e = [10,10,10,10,10,10,10,10,10,10]
    #print(GiniIndex(a))
    #print(HHI(a))
    print(GiniIndex(b))
    print(GiniIndex(c))
    print(GiniIndex(d))
    print(GiniIndex(e))
    print(HHI(b))
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")

