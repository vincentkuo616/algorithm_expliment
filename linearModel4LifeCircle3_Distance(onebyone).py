# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 10:28:30 2020

@author: vincentkuo
"""

import time
import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#import xgboost as xgb

loc="C://Users//vincentkuo//Documents//productLifeCircle"
dataName="//1349_products.xlsx"
#dataName="//0I388_example.xlsx"
#dataName="//test4RawData.xlsx"
#dataName="//1N400_pure_two.xlsx"
dataName="//1349_products_test.xlsx"
outputName="//1N400_pure_weekOutput_All.xlsx"
outputName4SF="//1349_productsOutput_All_4SF_Distance&Diff_05v2.xlsx"
saveFlag = 0
save4SFlag = 0
saveFlagLinear = 0
save4SFlagLinear = 0
groupMaxLength = 1000
dict4pk={'Z0207':[0,23,28,38,43,48,61,66,85],
         '0C450':[0,8,19,31,37],
         '1E836':[0,11,54,59,77],
         '1F517':[0,7,16,40,43],
         'Z8552':[0,10,18,33,42,50,54],
         '1H603':[0,15,28,55,87,98,114,119,125],
         '1N400':[0,10,42,48,62,70],
         'S9078':[0,8,39,45,57,104,110],
         'S7621':[0,10,25,32,38,47,52,57,82,97,102,111,123,128,147,155,160,165,205]}
dict4pk2={'Z0207':[0,29,38,43,48,61,66,85],
         '0C450':[0,8,22,30,37],
         '1E836':[0,12,62,77],
         '1F517':[0,8,16,43],
         'Z8552':[0,11,18,33,43,50,54],
         '1H603':[0,125],
         '1N400':[0,14,70],
         'S9078':[0,8,110],
         'S7621':[0,10,25,32,38,47,52,57,85,94,101,111,123,128,157,167,205]}
dict4pk3={'Z0207':[0,29,38,43,48,68,85],
         '0C450':[0,8,22,30,37],
         '1E836':[0,11,58,77],
         '1F517':[0,9,16,41,43],
         'Z8552':[0,11,18,54],
         '1H603':[0,125],
         '1N400':[0,14,42,48,62,70],
         'S9078':[0,8,110],
         'S7621':[0,10,205]}
dict4pk4={'Z0207':[0,29,38,68,85],
         '0C450':[0,8,22,30,37],
         '1E836':[0,11,58,77],
         '1F517':[0,9,43],
         'Z8552':[0,11,18,54],
         '1H603':[0,125],
         '1N400':[0,14,66,70],
         'S9078':[0,110],
         'S7621':[0,205]}
stageList=['Stage_' + str(i) for i in list(range(1,101))]

#np.seterr(divide='ignore',invalid='ignore') #忽略 NUMPY WARM
times = 3
timesList = [3,4,5,6,7]
outputName4SFList=["//1349_productsOutput_All_4SF_03v10.xlsx","//1349_productsOutput_All_4SF_Distance_03v10.xlsx","//1349_productsOutput_All_4SF_Combine_03v10.xlsx","//1349_productsOutput_All_4SF_Distance&Diff_03v10.xlsx","//1349_productsOutput_All_4SF_Default_03v10.xlsx"]
#timesList = [4,5,6,7]
#outputName4SFList=["//1349_productsOutput_All_4SF_Distance_03v9.xlsx","//1349_productsOutput_All_4SF_Combine_03v9.xlsx","//1349_productsOutput_All_4SF_Distance&Diff_03v9.xlsx","//1349_productsOutput_All_4SF_Default_03v9.xlsx"]
timesList = [4]
outputName4SFList=["//1349_productsOutput_All_4SF_03v9.xlsx"]

def main():
    for i in timesList:
        global times
        global outputName4SF
        times=i
        outputName4SF=outputName4SFList[timesList.index(i)]
        root = loc+dataName
        data = readData(root)
        print(data.head())
        # Null檢核
        check=0
        for i in data.values:
            if math.isnan(i[1]):
                check=1
        if check==1:
            print("check Start")
            data=processNullData(data)
        #outputData = gompertzLoop(data)
        
        outputDataByLinear = linearLoop(data)
        #第一版的輸出
        if saveFlag==1:
            outputData = np.hstack((np.transpose([data.columns.values]),outputData))
            saveData(outputData,loc,outputName,data.columns.values)
        # For Spotfire 的輸出
        if save4SFlag==1:
            save4SFdata = processData(outputData,data.columns.values)
            
            saveData(save4SFdata,loc,outputName4SF,data.columns.values)
        if saveFlagLinear==1:
            outputData = np.hstack((np.transpose([data.columns.values]),outputDataByLinear))
            
            saveData(outputData,loc,outputName,data.columns.values)
        if save4SFlagLinear==1:
            save4SFdata = processData(outputDataByLinear,data.columns.values)
            
            saveData(save4SFdata,loc,outputName4SF,data.columns.values)
        
        
    
def readData(root):
    try:
        if root[-3:]=='csv':
            data = pd.read_csv(root,encoding='utf-16')
        elif root[-3:]=='lsx':
            data = pd.read_excel(root,encoding='utf-8')
        else:
            data = pd.read_excel(root,encoding='utf-8')
    except:
        print("Data parsing failed")
    return data

def saveData(data,loc,outputName,columns):
    try:
        timestr = getDatetimeStr()
        outputName = outputName.split('.')[0]+'_'+timestr+'.'+outputName.split('.')[1]
        outputData = pd.DataFrame(data[1:],columns=data[0])
        if outputName.split('.')[1]=='xlsx':
            location = loc+outputName
            outputData.to_excel(location,index=False)
        elif outputName.split('.')[1]=='csv':
            outputData.to_csv(loc+outputName,index=False)
        else:
            print(outputData)
            
        print('Save Succeed')
    except:
        print("SaveData failed")

def gompertzLoop(data):
    try:
        dataArray = data.values
        outputArray=[]
        for i in dataArray:
            returnData = gompertzParameter_2(i)
            if len(outputArray)==0: outputArray=returnData
            else: outputArray = np.hstack((outputArray,returnData))
    except:
        print("gompertzLoop Error")
    return outputArray

def linearLoop(data):
    try:
        dataArray = data.values
        returnArray=[]
        for i in dataArray:
            dataAcc=cumulateData(i)
            dataY,dataX=[],[]
            index=1
            for j in dataAcc:
                if type(j)==str or math.isnan(j):
                    pass
                else:
                    dataX.append([index])
                    dataY.append(j)
                    index+=1
            group,slope,slope_inner = linearParameter(np.array(dataX),np.array(dataY),i[0])
            # 組CDF資料
            yCDF=[str(i[0])+'_CDF']
            for elements in dataY: yCDF.append(elements)
            while len(yCDF)<len(i):
                yCDF.append(math.nan)
            # 組分組資料
            yStage=[str(i[0])+'_Stage']
            for k in range(len(group)-1):
                index=group[k]
                while index<group[k+1]:
                    yStage.append(stageList[k])
                    index+=1
            while len(yStage)<len(i):
                yStage.append(math.nan)
            # 組分組資料(調整)
            yStageAdjusted=[str(i[0])+'_StageAdjusted']
            adjustedGroup=[]
            for g in range(len(group)):
                if g==0 or g==len(group)-1:
                    adjustedGroup.append(group[g])
                elif group[g]+3>group[len(group)-1]:
                    pass
                else:
                    adjustedGroup.append(group[g]+3)
            for k in range(len(adjustedGroup)-1):
                index=adjustedGroup[k]
                while index<adjustedGroup[k+1]:
                    yStageAdjusted.append(stageList[k])
                    index+=1
            while len(yStageAdjusted)<len(i):
                yStageAdjusted.append(math.nan)
            # 組警告訊號(調整 第二次才算)
            yWarningFlag=[str(i[0])+'_Warning']
            slopeTemp = ['N','N']
            if len(slope)>2:
                for u in range(2,len(slope)):
                    if slope[u]<0.1:
                        slopeTemp.append('WarnWarn')
                    elif slope_inner[u]<slope_inner[u-1] and slope[u]<0.3:
                        slopeTemp.append('Warn')
                    elif slope_inner[u]<slope_inner[u-1]:
                        slopeTemp.append('NWarn')
                    else:
                        slopeTemp.append('N')
            for w in range(len(adjustedGroup)-1):
                jumpFlag=0
                for ww in range(adjustedGroup[w],adjustedGroup[w+1]):
                    #前十二個不算
                    if ww<12:
                        yWarningFlag.append('Less')
                    else:
                        #訊號的處理
                        #Y200716 水平的處理
                        checkFlag=0
                        if adjustedGroup[w]>=3:
                            mean = (dataY[ww]-dataY[0])/(ww+1)
                            yTemp=dataY[adjustedGroup[w]-3:ww+1]
                            xTemp=dataX[adjustedGroup[w]-3:ww+1]
                            if mean==0 or LinearRegression().fit(xTemp,yTemp).coef_[0]/mean<0.1:
                                yWarningFlag.append('WarnWarn')
                                checkFlag=1
                            elif LinearRegression().fit(xTemp,yTemp).coef_[0]<slope_inner[w] and LinearRegression().fit(xTemp,yTemp).coef_[0]/mean<0.3:
                                yWarningFlag.append('Warn')
                                checkFlag=1
                            elif LinearRegression().fit(xTemp,yTemp).coef_[0]<slope_inner[w]:
                                yWarningFlag.append('NWarn')
                                checkFlag=1
                        #Y200716 group 跳的處理
                        if ww==adjustedGroup[w+1]-3:
                            if dataY[ww]-dataY[ww-1]>1.5*slope_inner[w+1] and checkFlag==1:
                                yWarningFlag[-1]='N'
                                jumpFlag=1
                            elif dataY[ww]-dataY[ww-1]>1.5*slope_inner[w+1]:
                                yWarningFlag.append('N')
                                checkFlag=1
                                jumpFlag=1
                        elif  checkFlag==0 and (ww==adjustedGroup[w+1]-2 or ww==adjustedGroup[w+1]-1):
                            if dataY[ww]-dataY[ww-1]>1.5*slope_inner[w+1] or jumpFlag==1:
                                yWarningFlag.append('N')
                                checkFlag=1
                        elif ww==adjustedGroup[w+1]-2 or ww==adjustedGroup[w+1]-1:
                            if dataY[ww]-dataY[ww-1]>1.5*slope_inner[w+1] or jumpFlag==1:
                                yWarningFlag[-1]='N'
                                jumpFlag=1
                        if checkFlag==0:
                            yWarningFlag.append(slopeTemp[w+1])
            while len(yWarningFlag)<len(i):
                yWarningFlag.append(math.nan)
            # 組警告訊號(第一次就看)
            yWarningSensitiveFlag=[str(i[0])+'_WarningSensitive']
            slopeTempSensitive = ['N','N']
            if len(slope)>2:
                for u in range(2,len(slope)):
                    if slope[u]<0.1:
                        slopeTempSensitive.append('WarnWarn')
                    elif slope_inner[u]<slope_inner[u-1] and slope_inner[u-1]<slope_inner[u-2] and u!=2:
                        slopeTempSensitive.append('WarnWarn')
                    elif slope_inner[u]<slope_inner[u-1] and slope[u]<0.3:
                        slopeTempSensitive.append('Warn')
                    elif slope_inner[u]<slope_inner[u-1]:
                        slopeTempSensitive.append('NWarn')
                    else:
                        slopeTempSensitive.append('N')
            for w in range(len(adjustedGroup)-1):
                for ww in range(adjustedGroup[w],adjustedGroup[w+1]):
                    yWarningSensitiveFlag.append(slopeTempSensitive[w+1])
            while len(yWarningSensitiveFlag)<len(i):
                yWarningSensitiveFlag.append(math.nan)
            
            returnArray.append(i.tolist())
            returnArray.append(yCDF)
            returnArray.append(yStage)
            returnArray.append(yStageAdjusted)
            returnArray.append(yWarningFlag)
            returnArray.append(yWarningSensitiveFlag)
            print(yWarningFlag)
    except:
        print("linearLoop Error")
    return np.transpose(returnArray)

def gompertzParameter(data):
    try:
        print('**********gompertzParameter**********')
        if len(data)<4:
            print('Not Enoungh Data to create Gompertz')
        else:
            pk = data[0]
            data4Compute = data[1:]
            outputList=[[pk,pk+'_b',pk+'_a',pk+'_k',pk+'_stage'],[1,0,0,0,0],[2,0,0,0,0]]
            for i in range(3,len(data4Compute)+1):
                if i%3==0:
                    length=i//3
                    one,two,three=0,0,0
                    for a in range(length):
                        one += math.log(data4Compute[a]+1)
                        two += math.log(data4Compute[a+length]+1)
                        three += math.log(data4Compute[a+2*length]+1)
                elif i%3==1:
                    length=i//3
                    one,two,three=0,0,0
                    for a in range(length):
                        one += math.log(data4Compute[a+1]+1)
                        two += math.log(data4Compute[a+length+1]+1)
                        three += math.log(data4Compute[a+2*length+1]+1)
                else:
                    length=i//3
                    one,two,three=0,0,0
                    for a in range(length):
                        one += math.log(data4Compute[a+2]+1)
                        two += math.log(data4Compute[a+length+2]+1)
                        three += math.log(data4Compute[a+2*length+2]+1)
                        
                outputList.append(calculateCenter(i,one,two,three))
                
            outputArray = np.transpose(np.array(outputList))
            outputArrayAndPK = np.vstack([np.array([data]),outputArray])
    except:
        print("gompertzParameter Error")
    return outputArrayAndPK

def gompertzParameter_2(data):
    try:
        print('**********gompertzParameter**********')
        if len(data)<4:
            print('Not Enoungh Data to create Gompertz')
        else:
            pk = str(data[0])
            data4Compute = data[1:]
            ''' 產出連續值 '''
            data4ComputeCDF = cumulateData(data)
            outputList=[[pk+'_b',pk+'_a',pk+'_k',pk+'_stage'],[0,0,0,0],[0,0,0,0]]
            for i in range(3,len(data4Compute)+1):
                length = lambda i:i//3 if i<(groupMaxLength+1)*3 else groupMaxLength
                if i%3==0:
                    one,two,three=0,0,0
                    for a in range(length(i)):
                        if i<(groupMaxLength+1)*3:
                            one += math.log(data4Compute[a]+1)
                            two += math.log(data4Compute[a+length(i)]+1)
                            three += math.log(data4Compute[a+2*length(i)]+1)
                        else:
                            one += math.log(data4Compute[i-a-1-2*groupMaxLength]+1)
                            two += math.log(data4Compute[i-a-1-1*groupMaxLength]+1)
                            three += math.log(data4Compute[i-a-1]+1)
                elif i%3==1:
                    one,two,three=0,0,0
                    for a in range(length(i)):
                        if i<(groupMaxLength+1)*3:
                            one += math.log(data4Compute[a+1]+1)
                            two += math.log(data4Compute[a+length(i)+1]+1)
                            three += math.log(data4Compute[a+2*length(i)+1]+1)
                        else:
                            one += math.log(data4Compute[i-a-1-2*groupMaxLength]+1)
                            two += math.log(data4Compute[i-a-1-1*groupMaxLength]+1)
                            three += math.log(data4Compute[i-a-1]+1)
                else:
                    one,two,three=0,0,0
                    for a in range(length(i)):
                        if i<(groupMaxLength+1)*3:
                            one += math.log(data4Compute[a+2]+1)
                            two += math.log(data4Compute[a+length(i)+2]+1)
                            three += math.log(data4Compute[a+2*length(i)+2]+1)
                        else:
                            one += math.log(data4Compute[i-a-1-2*groupMaxLength]+1)
                            two += math.log(data4Compute[i-a-1-1*groupMaxLength]+1)
                            three += math.log(data4Compute[i-a-1]+1)
                outputList.append(calculateCenter(length(i),one,two,three))
            
            pkArray = np.transpose(np.vstack([np.array([data]),np.array([data4ComputeCDF])]))
            outputArrayAndPK = np.hstack([pkArray,outputList])
    except:
        print("gompertzParameter_2 Error")
    return outputArrayAndPK

def linearParameter(dataX,dataY,pk):
    try:
        print('**********linearParameter**********')
        if len(dataX)<5:
            print('Not Enoungh Data to create Linear')
            group = [0,len(dataX)]
            slope = [0,len(dataX)]
            slope_inner = [0,len(dataX)]
        else:
            group = groupRule(dataX,dataY)
            #group = groupAdjust(group,dataX,dataY)
            slope = [0]
            slope_inner = [0]
            
            import matplotlib.pyplot as plt
            plt.cla()
            
            for i in range(0,len(group)-1):
                testX=dataX[group[i]:group[i+1]]
                testY=dataY[group[i]:group[i+1]]
                
                dataX4Model=dataX[group[i]:group[i+1]]
                dataY4Model=dataY[group[i]:group[i+1]]
                model = LinearRegression().fit(dataX4Model,dataY4Model)

                xfit=np.linspace(group[i],group[i+1],100)
                yfit=model.predict(xfit[:,np.newaxis])
                plt.scatter(testX,testY)
                plt.plot(xfit,yfit)
                slope_inner.append(model.coef_[0])
                #slope.append(model.coef_[0]/(testY[-1]/(group[i+1])))
                slope.append(model.coef_[0]/((testY[-1]-dataY[0])/(group[i+1])))
                '''
                timestr = getDatetimeStr()
                locate="C://Users//vincentkuo//Documents//productLifeCircle//plot//plot"+str(week)+timestr+".png"
                plt.savefig(locate)
                '''
            #plt.show()
    except:
        print("linearParameter Error")
    return group,slope,slope_inner

def groupAdjust(group,dataX,dataY):
    try:
        groupNew=[0]
        slope=-1
        maxNum=len(dataX)
        for i in range(len(group)-1):
            dataX4Model=dataX[group[i]:group[i+1]]
            dataY4Model=dataY[group[i]:group[i+1]]
            model = LinearRegression().fit(dataX4Model,dataY4Model)
            tempSlope = model.coef_[0]/(dataY4Model[-1]/(group[i+1]-1))
            slopeNew=tempSlope/math.sqrt(tempSlope**2+1)
            if abs(slopeNew-slope)<0.05:
                groupNew[-1]=group[i+1]
                slope=slopeNew
            else:
                groupNew.append(group[i+1])
                slope=slopeNew
    except:
        print("groupAdjust Error")
        groupNew.append(maxNum)
    return groupNew

def groupRule(dataX,dataY):
    try:
        group=[0]
        ruleNum,maxNum=len(dataX),len(dataX)
        while group[-1]<len(dataX):
            corrList=[]
            for i in range(group[-1]+5,maxNum+1):
                corrList.append(np.corrcoef(dataX[group[-1]:i].ravel(),dataY[group[-1]:i].ravel())[0,1])
            #後續計算距離用
            dataListY=dataY[group[-1]:maxNum]
            dataListX=dataX[group[-1]:maxNum]
            if len(corrList)>0: ruleNum=group[-1]+5+rule(corrList,dataListX,dataListY)
            else: ruleNum=maxNum
            group.append(ruleNum)
    except:
        print("groupRule Error")
        group.append(maxNum)
    return group

def rule(corrList,dataListX,dataListY):
    try:
        index=len(corrList)-1 #最後一個 (因為不等於取LENGTH 等於要減一)
        if times==1:
            for i in range(len(corrList)-1):
                defOfDown=0.002
                defOfDown=0.1005-0.1*corrList[i]
                if corrList[i]>corrList[i+1] and corrList[i]-corrList[i+1]>defOfDown:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.1:
                    index=i
                    break
        #連續兩次下降大於1.005-R的0.1倍
        if times==2:
            for i in range(len(corrList)-2):
                defOfDown=0.002
                defOfDown=0.1005-0.1*corrList[i]
                if corrList[i]>corrList[i+1] and corrList[i+1]>corrList[i+2] and corrList[i]-corrList[i+1]>defOfDown and corrList[i+1]-corrList[i+2]>defOfDown:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.5/(i+5) and corrList[i]-corrList[i+2]<0.07:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.03*(i+5):
                    index=i
                    break
        #'''
        #DEFAULT
        if times==7:
            for i in range(len(corrList)-3):
                defOfDown=0
                if corrList[i]>corrList[i+1] and corrList[i+1]>corrList[i+2] and corrList[i+2]>corrList[i+3] and corrList[i]-corrList[i+1]>defOfDown and corrList[i+1]-corrList[i+2]>defOfDown and corrList[i+2]-corrList[i+3]>defOfDown:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.2:
                    index=i
                    break
        #'''
        if times==3: #方法一
            for i in range(len(corrList)-3):
                defOfDown=0.002
                defOfDown=(1.09-corrList[i])*0.5/(i+5)
                dataX=dataListX[0:i+5]
                dataY=dataListY[0:i+5]
                if corrList[i]>corrList[i+1] and corrList[i+1]>corrList[i+2] and corrList[i+2]>corrList[i+3] and corrList[i]-corrList[i+1]>defOfDown and corrList[i+1]-corrList[i+2]>defOfDown and corrList[i+2]-corrList[i+3]>defOfDown:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.5/(i+5) and corrList[i]-corrList[i+2]<0.07:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.03*(i+5):
                    index=i
                    break
        if times==4: #方法二
            for i in range(len(corrList)-3):
                defOfDown=0.002
                defOfDown=(1.09-corrList[i])*0.5/(i+5)
                increment=1
                dataX=dataListX[0:i+5]
                dataY=dataListY[0:i+5]
                model = LinearRegression().fit(dataX,dataY)
                mean = dataY[-1]/(i+5)
                first = model.predict([dataListX[i+5]])
                second = model.predict([dataListX[i+6]])
                third = model.predict([dataListX[i+7]])
                firstDiff = abs(first-dataListY[i+5])
                secondDiff = abs(second-dataListY[i+6])
                thirdDiff = abs(third-dataListY[i+7])
                if first>dataListY[i+5] and second>dataListY[i+6] and third>dataListY[i+7] and firstDiff*increment<secondDiff and secondDiff*increment<thirdDiff:
                    index=i
                    break
                if first<dataListY[i+5] and second<dataListY[i+6] and third<dataListY[i+7] and firstDiff*increment<secondDiff and secondDiff*increment<thirdDiff:
                    index=i
                    break
        if times==5: #方法三
            for i in range(len(corrList)-3):
                defOfDown=0.002
                defOfDown=(1.09-corrList[i])*0.5/(i+5)
                increment=1.5
                dataX=dataListX[0:i+5]
                dataY=dataListY[0:i+5]
                model = LinearRegression().fit(dataX,dataY)
                mean = dataY[-1]/(i+5)
                first = model.predict([dataListX[i+5]])
                second = model.predict([dataListX[i+6]])
                third = model.predict([dataListX[i+7]])
                firstDiff = abs(first-dataListY[i+5])
                secondDiff = abs(second-dataListY[i+6])
                thirdDiff = abs(third-dataListY[i+7])
                if corrList[i]>corrList[i+1] and corrList[i+1]>corrList[i+2] and corrList[i+2]>corrList[i+3] and corrList[i]-corrList[i+1]>defOfDown and corrList[i+1]-corrList[i+2]>defOfDown and corrList[i+2]-corrList[i+3]>defOfDown:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.5/(i+5) and corrList[i]-corrList[i+2]<0.07:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.03*(i+5):
                    index=i
                    break
                if first>dataListY[i+5] and second>dataListY[i+6] and third>dataListY[i+7] and firstDiff*increment<secondDiff and secondDiff*increment<thirdDiff:
                    index=i
                    break
                if first<dataListY[i+5] and second<dataListY[i+6] and third<dataListY[i+7] and firstDiff*increment<secondDiff and secondDiff*increment<thirdDiff:
                    index=i
                    break
                if dataListY[i+5]==dataListY[i+6] and dataListY[i+6]==dataListY[i+7] and model.coef_[0]>=0.2:
                    index=i
                    break
                if firstDiff>model.coef_[0]*1.5+3:
                    index=i
                    break
        if times==6: #方法四
            for i in range(len(corrList)-3):
                defOfDown=0.002
                defOfDown=(1.09-corrList[i])*0.5/(i+5)
                increment=2
                dataX=dataListX[0:i+5]
                dataY=dataListY[0:i+5]
                model = LinearRegression().fit(dataX,dataY)
                mean = dataY[-1]/(i+5)
                first = model.predict([dataListX[i+5]])
                second = model.predict([dataListX[i+6]])
                third = model.predict([dataListX[i+7]])
                firstDiff = abs(first-dataListY[i+5])
                secondDiff = abs(second-dataListY[i+6])
                thirdDiff = abs(third-dataListY[i+7])
                if first>dataListY[i+5] and second>dataListY[i+6] and third>dataListY[i+7] and firstDiff*increment<secondDiff and secondDiff*increment<thirdDiff:
                    index=i
                    break
                if first<dataListY[i+5] and second<dataListY[i+6] and third<dataListY[i+7] and firstDiff*increment<secondDiff and secondDiff*increment<thirdDiff:
                    index=i
                    break
                if firstDiff>mean:
                    index=i
                    break
        
    except:
        print('Rule Calculate Error')
    return index

def calculateCenter(index,one,two,three):
    try:
        p_b,p_a,p_k=0,0,0
        p_n=index
        if (three-two)*(two-one)<=0 or math.isnan(three):
            pass
        else:
            p_b = math.pow(((three-two)/(two-one)),1/p_n)
            p_a = math.exp((p_b-1)*(two-one)/(math.pow((math.pow(p_b,p_n)-1),2)))
            p_k = math.exp((one-math.log(p_a)*(math.pow(p_b,p_n)-1)/(p_b-1))/p_n)
    except:
        print("calculateCenter Error")
    return [p_b,p_a,p_k,defineStage(p_b,p_a)]

def defineStage(b,a):
    try:
        stage='轉折'
        if b>1 and a>1:
            stage='導入期'
        elif b<1 and b>0 and a<1 and a>0:
            stage='成長期'
        elif b>1 and a<1 and a>0:
            stage='衰退前期'
        elif a>1 and b>0 and b<1:
            stage='衰退後期'
    except:
        print("defineStage Error")
    return stage

def defineStageByLinear(week,model,dataX,dataY):
    try:
        stage='轉折'
        yHat = model.predict(dataX)
        diff = dataY-yHat
        if (diff<0).all() and (np.sort(diff)[::-1]==diff).all():
            stage='警告'
    except:
        print("defineStageByLinear Error")
    return stage

def getDatetimeStr():
    d = datetime.now()
    str='{:%Y%m%d%H%M%S}'.format(d)
    return str

'''
將資料從null null data data null
轉變成  data data null null null
'''
def processNullData(data):
    try:
        columns=[data.columns.values[0]]
        data=data.values
        dataClearNull,dataOutput=[],[]
        for i in data:
            pk=i[0]
            data4Loop=[]
            data4Loop.append(pk)
            data4Compute = i[1:]
            for i in range(len(data4Compute)):
                if math.isnan(data4Compute[i]):
                    pass
                else:
                    data4Loop.append(data4Compute[i])
            dataClearNull.append(data4Loop)
        maxLength=0
        # 找最大長度
        for j in dataClearNull:
            if maxLength<len(j):
                maxLength=len(j)
        # 補資料到最大長度
        for k in dataClearNull:
            while len(k)<maxLength:
                k.append(math.nan)
            dataOutput.append(k)
        # 組 Columns
        for l in range(1,maxLength):
            columns.append(l)
        pandasReturn=pd.DataFrame(dataOutput,columns=np.array(columns))
        
    except:
        print("ProcessNullData Error")
    return pandasReturn

def processData(data,columns):
    try:
        length = len(data[0])//6
        outputList=[['WEEK','ProductName','CDF','STAGE','STAGEADJUSTED','WARNFLAG','WARNFLAG_SENSITIVE']]
        for i in range(length):
            one=data[1:,i*6+1]
            two=data[1:,i*6+2]
            three=data[1:,i*6+3]
            four=data[1:,i*6+4]
            five=data[1:,i*6+5]
            productName=data[0,i*6]
            for j in range(len(one)):
                if one[j]=='nan': pass
                else: outputList.append([j+1,productName,one[j],two[j],three[j],four[j],five[j]])
        
    except:
        print("ProcessData Error")
    return np.array(outputList)

def cumulateData(data):
    try:
        pk = data[0]
        data4Compute = data[1:]
        ''' 產出連續值 '''
        dataCDF = [str(pk)+'_CDF']
        for i in range(len(data4Compute)):
            if math.isnan(data4Compute[i]):
                dataCDF.append(math.nan)
            else:
                if i==0:    dataCDF.append(0+data4Compute[i])
                else:   dataCDF.append(dataCDF[-1]+data4Compute[i])
    except:
        print('CumulateData Error')
    return dataCDF

if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")