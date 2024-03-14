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
dataName="//1N400_pure_two.xlsx"
outputName="//1N400_pure_weekOutput_All.xlsx"
outputName4SF="//1N400_pure_weekOutput_All_4SF.xlsx"
saveFlag = 0
save4SFlag = 0
groupMaxLength = 1000

def main():
    root = loc+dataName
    data = readData(root)
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
            linearParameter(np.array(dataX),np.array(dataY))
        outputArray=[1]
    except:
        print("linearLoop Error")
    return outputArray

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
            pk = data[0]
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

def linearParameter(dataX,dataY):
    try:
        print('**********linearParameter**********')
        if len(dataX)<5:
            print('Not Enoungh Data to create Linear')
        else:
            model4Week=[]
            for i in range(5,len(dataX)+1):
                week=i+1
                testX=dataX[0:week]
                testY=dataY[0:week]
                rSquared=0
                modelBest = LinearRegression()
                stage=''
                for j in range(5,week):
                    for k in range(0,week-j):
                        dataX4Model=dataX[k:k+j]
                        dataY4Model=dataY[k:k+j]
                        model = LinearRegression().fit(dataX4Model,dataY4Model)
                        '''
                        #MAE
                        if rSquared > mean_absolute_error(np.array(testY),model.predict(testX)):
                            modelBest= model
                            rSquared = mean_absolute_error(np.array(testY),model.predict(testX))
                        '''
                        if rSquared < model.score(np.array(testX),np.array(testY)):
                            modelBest= model
                            rSquared = model.score(np.array(testX),np.array(testY))
                        #'''
                #model4Week.append([week,modelBest])
                if week==len(dataX):  stage=defineStageByLinear(week,modelBest,dataX[week:week+1],dataY[week:week+1])
                else:   stage=defineStageByLinear(week,modelBest,dataX[week:week+2],dataY[week:week+2])
                #'''
                import matplotlib.pyplot as plt
                xfit=np.linspace(0,i,100)
                yfit=modelBest.predict(xfit[:,np.newaxis])
                plt.cla()
                plt.scatter(testX,testY)
                plt.plot(xfit,yfit)
                timestr = getDatetimeStr()
                #'''
                locate="C://Users//vincentkuo//Documents//productLifeCircle//plot//plot"+str(week)+timestr+".png"
                plt.savefig(locate)
                #'''
                model4Week.append([week,stage,rSquared,modelBest.coef_[0]])
            print(model4Week)
                    
            outputArrayAndPK = []
    except:
        print("linearParameter Error")
    return outputArrayAndPK

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

def processData(data,columns):
    try:
        length = len(data[0])//6
        outputList=[['WEEK','ProductName','CDF','STAGE']]
        for i in range(length):
            one=data[1:,i*6+1]
            two=data[1:,i*6+5]
            productName=data[0,i*6]
            for j in range(len(one)):
                if one[j]=='nan': pass
                else: outputList.append([j+1,productName,one[j],two[j]])
        
    except:
        print("ProcessData Error")
    return np.array(outputList)

def cumulateData(data):
    try:
        pk = data[0]
        data4Compute = data[1:]
        ''' 產出連續值 '''
        dataCDF = [pk+'_CDF']
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