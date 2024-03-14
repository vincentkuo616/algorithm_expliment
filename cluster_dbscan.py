# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:46:12 2022

@author: vincentkuo
"""


import numpy as np
import pandas as pd 
import time
from datetime import datetime
from sklearn.cluster import DBSCAN
from collections import Counter

prjRootFolder = "C:\\vincent\\AI_DBS\\GP\\python"
prjRootFolder = "C:\\vincent\\AI_MIS\\公司員工習性分析\\python"
prjRootFolder = "D:\\文件"

#prjRootFolder = "C:\\Users\\syn009\\Documents\\"
dataName = '\\data\\allCust_csg_CN_xFOB(BIN).xlsx'
#dataName = '4dbscan_3.xlsx'
dataName = '\\1424_noRead.xlsx'
dataName = '\\1424_Read.xlsx'
dataName = '\\1365_noRead.xlsx'
dataName = '\\1365_Read.xlsx'
dataName = '\\金額_4DBSCAN.xlsx'
dataName = '\\4Kmeans(cmr).xlsx'
#dataName = '\\金額_4DBSCAN2.csv'
#dataName = '\\1424_noRead_2.xlsx'
outputName = "\\CN\\.xlsx"
outputName = "\\.xlsx"
saveFlag = 1
features = ['PSCNT_LHY', 'PSAMT_TOP1RATIO_LHY', 'PSAMT_GINI_LHY', 'GGDATE_RATIO_LHY', 'GGAMT_MED_LHY', 'GGCNT_70PERCENT_LHY']
features = ['PSCNT_LHY', 'PSAMT_TOP1RATIO_LHY', 'PSAMT_GINI_LHY', 'GGDATE_RATIO_LHY', 'GGCNT_70PERCENT_LHY']
features = ['SALESAMT', 'GP_SPR_P']
features = ['GP_SPR_P']
features = ['noReadRate']
features = ['AMT'] #季度	報支單數	報支總額(NTD)	費用率	人員角色
features = ['報支總額(NTD)','報支單數'] #季度	報支單數		費用率	人員角色
features = ['報支總額(NTD)','費用率'] #季度	報支單數		費用率	人員角色
features = ['SALESAMT','OFFICEAMT','GGCNT','BrandCNT','CSGCNT','PSCNT'] #
features = ['SALESAMT','OFFICEAMT','GGCNT','BrandCNT'] #
#features = ['noReadRate2']
temp=""

def main():
    dataRoot = prjRootFolder+dataName
    data = readData(dataRoot)
    print(data.head())
    
#    data=data.iloc[62:1271,:]
#    dataGroup=data.groupby('CS_GROUP')
    #data=data.loc[data['業績總額']>1000000]
    data=data.iloc[:,:]
    #dataGroup=data.groupby('人員角色')
    data4Save = pd.DataFrame()

    '''
    for key in dataGroup.groups.keys():
        dataSelected = dataGroup.get_group(key)
    
        X = dataSelected[features]
        normalized_X = (X-X.mean())/X.std()
        #設定每群最小要有5%以上的樣本
        leng = 1
        #設定每群的間距最小不能超過
        e = 2
        
        model = DBSCAN(min_samples=leng , eps=e).fit(normalized_X)
        labels2 = model.fit_predict(normalized_X)
        dataSelected.insert(2,'dbscan_cust',labels2)
        data4Save = data4Save.append(dataSelected, ignore_index = True)
        
    for key in dataGroup.groups.keys():
        dataSelected = dataGroup.get_group(key)
        dataBIN = dataSelected.groupby('季度')
        for k in dataBIN.groups.keys():
            dataBINSelected = dataBIN.get_group(k)
            print(dataBINSelected)
        
            X = dataBINSelected[features]
            normalized_X = (X-X.mean())/X.std()
            #設定每群最小要有5%以上的樣本
            leng = 5
            #設定每群的間距最小不能超過
            e = 0.5
            
            model = DBSCAN(min_samples=leng , eps=e).fit(normalized_X)
            labels2 = model.fit_predict(normalized_X)
            dataBINSelected.insert(2,'dbscan_金額_2',labels2)
            data4Save = data4Save.append(dataBINSelected, ignore_index = True)
            global temp
            temp = data4Save
    '''
    print(data)
        
    X = data[features]
    normalized_X = (X-X.mean())/X.std()
    #設定每群最小要有5%以上的樣本
    leng = 5
    #設定每群的間距最小不能超過
    e = 0.5
    
    model = DBSCAN(min_samples=leng , eps=e).fit(normalized_X)
    labels2 = model.fit_predict(normalized_X)
    data.insert(2,'dbscan_金額_2',labels2)
    data4Save = data4Save.append(data, ignore_index = True)
    global temp
    temp = data4Save
    #display first five rows of dataframe
    if saveFlag == 1:
        saveData(data4Save, 'dbscan_allCust1_csg_CN_xFOB(BIN)')
    else:
        print(data4Save.head())
    
def readData(root):
    try:
        if root[-3:]=='csv':
            data = pd.read_csv(root,encoding='utf-8')
        elif root[-3:]=='lsx':
            data = pd.read_excel(root)
        elif root[-3:]=='txt':
            data = pd.read_csv(root,header=None,encoding='utf-8')
        else:
            data = pd.read_excel(root,encoding='utf-8')
    except:
        print("Data parsing failed")
    return data

def saveData(data,dataName):
    try:
        timestr = getDatetimeStr()
        outputNameFinal = outputName.split('.')[0]+dataName+'_'+timestr+'.'+outputName.split('.')[1]
        outputData = data
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

def getDatetimeStr():
    d = datetime.now()
    str='{:%Y%m%d%H%M%S}'.format(d)
    return str

if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
