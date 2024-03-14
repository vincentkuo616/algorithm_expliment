# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:52:07 2020

@author: vincentkuo
"""

#import numpy as np
import pandas as pd
import time 

def main():
    root = "C://Users//vincentkuo//Documents//corr_test.xlsx"
    data = readData(root)
    boundary=0.3
    featureNameList,featureNameIndex = deleteCorrelationThroughIV(data,boundary)
    print('FEATURE NAME: ',featureNameList)
    print('FEATURE INDEX: ',featureNameIndex)

def readData(root):
    try:
        if root[-3:]=='csv':
            data = pd.read_csv(root,encoding='utf-8')
        elif root[-3:]=='lsx':
            data = pd.read_excel(root,encoding='utf-8')
        else:
            data = pd.read_excel(root,encoding='utf-8')
    except:
        print("Data parsing failed")
    return data

def checkData4DeleteCorrelationThroughIV(data):
    try:
        featureName = data.columns.values[1:].tolist()
        corr_array = data.iloc[:,1:].values
        if len(featureName)!=len(corr_array[0]):
            print('data length exception')
        for i in range(0,len(featureName)):
            if corr_array[i][i]!=1:
                print('correlation data error index',i)
    except:
        print("checkData4DeleteCorrelationThroughIV failed")
    return corr_array,featureName

def deleteCorrelationThroughIV(data,boundary):
    try:
        corr_array,featureNameList = checkData4DeleteCorrelationThroughIV(data)
        
        #刪除邏輯
        removeList=[]
        removeIndexList=[]
        featureNameIndex=list(range(len(featureNameList)))
    
        for i in range(0,len(featureNameList)):
            for j in range (i+1,len(featureNameList)):
                if featureNameList[j] in removeList:
                    break
                elif corr_array[i][j]>boundary or corr_array[i][j]<-boundary:
                    removeList.append(featureNameList[j])
                    removeIndexList.append(j)
        for i in removeList: featureNameList.remove(i)
        for i in removeIndexList: featureNameIndex.remove(i)
    except:
        print('deleteCorrelationThroughIV failed')
    
    return featureNameList,featureNameIndex

if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")