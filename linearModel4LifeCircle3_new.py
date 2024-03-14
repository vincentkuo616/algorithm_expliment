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

loc="C://Users//vincentkuo//Documents//productLifeCircle"
dataName="//1349_products.xlsx"
#dataName="//1349_products_test.xlsx"
outputName="//1N400_pure_weekOutput_All.xlsx"
outputName4SF="//1349_productsOutput_All_4SF_Distance&Diff_05v2.xlsx"
saveFlagLinear = 0
save4SFlagLinear = 1
groupMaxLength = 1000
stageList=['Stage_' + str(i) for i in list(range(1,101))]

#np.seterr(divide='ignore',invalid='ignore') #忽略 NUMPY WARM
times = 3
timesList = [5]
outputName4SFList=["//temp.xlsx"]

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
        
        outputDataByLinear = linearLoop(data)
        #第一版的輸出
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
                            '''
                            elif LinearRegression().fit(xTemp,yTemp).coef_[0]<slope_inner[w] and LinearRegression().fit(xTemp,yTemp).coef_[0]/mean<0.3:
                                yWarningFlag.append('Warn')
                                checkFlag=1
                            elif LinearRegression().fit(xTemp,yTemp).coef_[0]<slope_inner[w]:
                                yWarningFlag.append('NWarn')
                                checkFlag=1
                            '''
                            if any(LinearRegression().fit(xTemp,yTemp).coef_[0]<s for s in slope_inner[0:w+1]) and LinearRegression().fit(xTemp,yTemp).coef_[0]/mean<0.3 and checkFlag==0:
                                yWarningFlag.append('Warn')
                                checkFlag=1
                            if LinearRegression().fit(xTemp,yTemp).coef_[0]<slope_inner[w] and checkFlag==0:
                                yWarningFlag.append('NWarn')
                                checkFlag=1
                            elif checkFlag==0:
                                yWarningFlag.append('N')
                                checkFlag=1
                            '''
                            #Y200716 group 跳的處理
                            model = LinearRegression().fit(xTemp[:-1],yTemp[:-1])
                            if ww==adjustedGroup[w+1]-3:
                                if dataY[ww]-model.predict([dataX[ww]])>1.5*model.coef_[0]+13 and checkFlag==1:
                                    yWarningFlag[-1]='N'
                                    jumpFlag=1
                                elif dataY[ww]-model.predict([dataX[ww]])>1.5*model.coef_[0]+13:
                                    yWarningFlag.append('N')
                                    checkFlag=1
                                    jumpFlag=1
                            elif  checkFlag==0 and (ww==adjustedGroup[w+1]-2 or ww==adjustedGroup[w+1]-1):
                                if dataY[ww]-model.predict([dataX[ww]])>1.5*model.coef_[0]+13 or jumpFlag==1:
                                    yWarningFlag.append('N')
                                    checkFlag=1
                            elif ww==adjustedGroup[w+1]-2 or ww==adjustedGroup[w+1]-1:
                                if dataY[ww]-model.predict([dataX[ww]])>1.5*model.coef_[0]+13 or jumpFlag==1:
                                    yWarningFlag[-1]='N'
                                    jumpFlag=1
                            '''
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
    except:
        print("linearLoop Error")
    return np.transpose(returnArray)

def linearParameter(dataX,dataY,pk):
    try:
        print('**********linearParameter**********')
        if len(dataX)<4:
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

'''
起初為了讓斜率相近的Group合併
'''
def groupRule(dataX,dataY):
    try:
        group=[0]
        ruleNum,maxNum=len(dataX),len(dataX)
        while group[-1]<len(dataX):
            corrList=[]
            for i in range(group[-1]+4,maxNum+1):
                corrList.append(np.corrcoef(dataX[group[-1]:i].ravel(),dataY[group[-1]:i].ravel())[0,1])
            #後續計算距離用
            dataListY=dataY[group[-1]:maxNum]
            dataListX=dataX[group[-1]:maxNum]
            if len(corrList)>0: ruleNum=group[-1]+4+rule(corrList,dataListX,dataListY,group[-1],dataY[0])
            else: ruleNum=maxNum
            group.append(ruleNum)
    except:
        print("groupRule Error")
        group.append(maxNum)
    return group

def rule(corrList,dataListX,dataListY,groupValue,firstY):
    try:
        index=len(corrList)-1 #最後一個 (因為不等於取LENGTH 等於要減一)
        if times==5: #方法三
            for i in range(len(corrList)-3):
                defOfDown=0.002
                defOfDown=(1.09-corrList[i])*0.5/(i+5)
                increment=1.5
                dataX=dataListX[0:i+4]
                dataY=dataListY[0:i+4]
                model = LinearRegression().fit(dataX,dataY)
                mean = (dataY[-1]-firstY)/(i+4+groupValue)
                first = model.predict([dataListX[i+4]])
                second = model.predict([dataListX[i+5]])
                third = model.predict([dataListX[i+6]])
                firstDiff = abs(first-dataListY[i+4])
                secondDiff = abs(second-dataListY[i+5])
                thirdDiff = abs(third-dataListY[i+6])
                if corrList[i]>corrList[i+1] and corrList[i+1]>corrList[i+2] and corrList[i+2]>corrList[i+3] and corrList[i]-corrList[i+1]>defOfDown and corrList[i+1]-corrList[i+2]>defOfDown and corrList[i+2]-corrList[i+3]>defOfDown:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.5/(i+5) and corrList[i]-corrList[i+2]<0.07:
                    index=i
                    break
                elif corrList[i]-corrList[i+1]>0.03*(i+5):
                    index=i
                    break
                if first>dataListY[i+4] and second>dataListY[i+5] and third>dataListY[i+6] and firstDiff*increment<secondDiff and secondDiff*increment<thirdDiff:
                    index=i
                    break
                if first<dataListY[i+4] and second<dataListY[i+5] and third<dataListY[i+6] and firstDiff*increment<secondDiff and secondDiff*increment<thirdDiff:
                    index=i
                    break
                if dataListY[i+4]==dataListY[i+5] and dataListY[i+5]==dataListY[i+6] and model.coef_[0]/mean>=0.3:
                    index=i
                    break
                if firstDiff>model.coef_[0]*1.5+3:
                    index=i
                    break        
    except:
        print('Rule Calculate Error')
    return index

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