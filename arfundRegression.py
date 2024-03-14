# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:17:32 2020

@author: vincentkuo
"""

import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
from sklearn.linear_model import LinearRegression
#import xgboost as xgb
from sklearn import preprocessing
import pandas as pd
from collections import Counter
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
from scipy import stats


# 基礎資料設定
dataFolder = "C:\\Users\\vincentkuo\\Documents\\vincent_ARFund\\Data"
dataName   = "Customer_SanYu.xlsx"
dataName   = "All_Customer_V4.0_NoOutlier.xlsx"
dataName   = "All_Customer_V4.0_NoOutlier_ByGroup.csv"
dataName   = "All_Customer_V4.0_NoOutlier_ByGroup(only3).xlsx"
pk         = [2,3,4]
feature    = [6,7,27,40] #額度使用率(%) 月匯款次數 AR到期金額 月份
Y          = [11]
group      = ["額度不足"] #賺取利息 額度不足
save       = 0

def main():
    
    #cV=customerValue("", "", np.array(object), "", {}, {}, {}, {}, {},  0)
    
    dataFrame  = readData(dataFolder, dataName)
    print(dataFrame.head())
    dataFrame = dataFrame.fillna("NA")

    #dataFrame_credit = dataFrame.groupby("行為類型\n(Model_V104)").get_group("額度不足")
    dataFrame_credit = dataFrame.groupby("Model_V104").get_group(group[0])
    #dataFrame_credit = dataFrameModel_V104
    #print(dataFrame_credit.head())
    
    #dataFrame_credit = dataFrame.groupby("行為類型\n(Model_V104)").get_group("賺取利息")
    #dataFrame_credit = dataFrame.groupby("Model_V104").get_group("賺取利息")
    #print(dataFrame_rebate.head())
    
    '''
        Title處理 (Output-1)
    '''
    customerBehavior = np.array(["行為"])
    pkColumnName = dataFrame.iloc[:,pk].columns.values
    featuresColumnName = dataFrame.iloc[:,feature].columns.values
    featuresColumnNameMean = [elements+"_Mean" for elements in featuresColumnName]
    featuresColumnNameVar = [elements+"_Std" for elements in featuresColumnName]
    featuresColumnNameCoef = [elements+"_Coef" for elements in featuresColumnName]
    additionColumnName = np.array(["常數","MAE","MSE","MAXERROR","筆數","R2","R2_Adjusted","R2_Predicted","MaxVIF"])
    featuresColumnNameCorr = [elements+"_Corr" for elements in featuresColumnName]
    ColumnsName = np.concatenate((customerBehavior,pkColumnName,featuresColumnNameMean,featuresColumnNameVar,featuresColumnNameCoef,additionColumnName,featuresColumnNameCorr),axis=0)
    output = np.array([ColumnsName])
    '''
        Title處理 (Output-2)
    '''
    yActualColumnName = dataFrame.iloc[:,Y].columns.values
    yPredictColumnName = [elements+"_Predict" for elements in yActualColumnName]
    ColumnsNameDtl = np.concatenate((customerBehavior,pkColumnName,yActualColumnName,yPredictColumnName),axis=0)
    outputDtl = np.array([ColumnsNameDtl])
    
    outputCopy = output.copy()
    outputDtlCopy = outputDtl.copy()
    
    for i in range(0,len(list(Counter(dataFrame_credit.iloc[:,4])))):
        customerCodeAbbname = list(Counter(dataFrame_credit.iloc[:,4]))[i]
        data4LR = dataFrame_credit.groupby("交易客戶簡稱").get_group(customerCodeAbbname)
        outputCopy,outputDtlCopy = calculateCoef(data4LR,data4LR.shape[0],len(ColumnsName),outputCopy,outputDtlCopy)
        
    outputData = pd.DataFrame(outputCopy, columns=outputCopy[0]).drop([0])
    outputDataDtl = pd.DataFrame(outputDtlCopy, columns=outputDtlCopy[0]).drop([0])
    print(outputData)
    print(outputDataDtl)
    
    if save==1:
        outputData.to_csv("C:\\Users\\vincentkuo\\Documents\\vincent_ARFund\\Data\\output_credit_V4.3_NoOutlier_ByGroup.csv",index=False,encoding='utf-16')
        outputDataDtl.to_csv("C:\\Users\\vincentkuo\\Documents\\vincent_ARFund\\Data\\outputDtl_credit_V4.3_NoOutlier_ByGroup.csv",index=False,encoding='utf-16')
        #outputData.to_csv("C:\\Users\\vincentkuo\\Documents\\vincent_ARFund\\Data\\temp.csv",index=False,encoding='utf-16')
        #outputDataDtl.to_csv("C:\\Users\\vincentkuo\\Documents\\vincent_ARFund\\Data\\tempdtl.csv",index=False,encoding='utf-16')
    
    '''
    params = {
        'booster':'gblinear',
        'objective':'reg:linear',
        'silents':1,
        'nthread':-1,
        'lambda':2,
        'alpha':0,
        'eval_metric':'mae'
    }
    plst = params.items()
    dtrain = xgb.DMatrix(features, repayRate)
    model2 = xgb.train(plst, dtrain, 1000)
    print(model2.predict(dtrain))
    '''
    
def calculateCoef(data, dataNum, outputLength, output, outputDtl):
    try:
        featuresName = data.iloc[:,feature].columns.values
        pkValue = data.iloc[0,pk].values
        featuresValue = data.iloc[:,feature].values
        repayRate= data.iloc[:,Y].values
        pkAndY = pk.copy()+Y
        pkValueDtl = data.iloc[:,pkAndY].values
        
        if dataNum==1:
            numToAppend = outputLength-len(pkValue)-1
            if group[0]=='額度不足':
                groupArray=np.array(['額度不足'])
            else:
                groupArray=np.array(['賺取利息'])
            outputCoef = np.concatenate((groupArray,pkValue,[999]*numToAppend),axis=0)
            output = np.append(output,[outputCoef],axis=0)
            
            outputPredict = np.concatenate((groupArray,pkValueDtl[0],[999]),axis=0)
            outputDtl = np.append(outputDtl,[outputPredict],axis=0)
        else:
            if group[0]=='額度不足':
                groupArray=np.array(['額度不足'])
                groupArrayDtl=np.array([np.array(['額度不足']*dataNum)]).transpose()
            else:
                groupArray=np.array(['賺取利息'])
                groupArrayDtl=np.array([np.array(['賺取利息']*dataNum)]).transpose()
            
            # 標準化處理
            zscore = preprocessing.StandardScaler()
            features_zs = zscore.fit_transform(featuresValue)
            # 線性回歸預測
            model = LinearRegression()
            model.fit(features_zs,repayRate)
            
            # 統計量計算
            max_VIF = np.array(['Data less than 10'])
            featuresCorrWithY = np.array(['Data less than 10']*len(feature))
            if dataNum>=10:
                features4Corr = np.append(features_zs,repayRate,axis=1)
                print(pd.DataFrame(features4Corr).corr())
                featuresCorrWithY = pd.DataFrame(features4Corr).corr().values[-1,:-1]
                max_VIF = np.array([checkVIF(features_zs,featuresName)])
            '''
                誤差的計算
            '''
            mae = np.array([mean_absolute_error(repayRate,model.predict(features_zs))])
            mse = np.array([mean_squared_error(repayRate,model.predict(features_zs))])
            maxerror = np.array([max_error(repayRate,model.predict(features_zs))])
            rSquared = model.score(features_zs,repayRate)
            if (features_zs.shape[0]-features_zs.shape[1]-1)>0:
                rSquaredAdjust = 1 - (1 - rSquared) * ((features_zs.shape[0]-1) / (features_zs.shape[0]-features_zs.shape[1]-1))
            else:
                rSquaredAdjust = 'Data less than features'
            count = np.array([dataNum])
            rSquared = np.array([rSquared])
            rSquaredAdjust = np.array([rSquaredAdjust])
            rSquaredPredicted = np.array([predicted_r2(repayRate,model.predict(features_zs),features_zs)])
            
            print(mae,mse,maxerror,rSquared,rSquaredPredicted,count)
            
            outputCoef = np.concatenate((groupArray,pkValue,zscore.mean_,np.sqrt(zscore.var_),model.coef_[0],model.intercept_,mae,mse,maxerror,count,rSquared,rSquaredAdjust,rSquaredPredicted,max_VIF,featuresCorrWithY),axis=0)
            output = np.append(output,[outputCoef],axis=0)
            
            actualPredict = np.append(groupArrayDtl,pkValueDtl,axis=1)
            actualPredict = np.append(actualPredict,model.predict(features_zs),axis=1)
            outputDtl = np.append(outputDtl,actualPredict,axis=0)
    except:
        print("calculateCoef Error")
    return output,outputDtl

def readData(dataFolder, dataName):
    try:
        dataPath = dataFolder+"\\"+dataName
        if dataPath[-3:]=='csv':
            data = pd.read_csv(dataPath,encoding='utf-8')
        elif dataPath[-3:]=='lsx':
            data = pd.read_excel(dataPath,encoding='utf-8')
        else:
            data = pd.read_excel(dataPath,encoding='utf-8')
    except:
        print("Data parsing failed")
    return data

def checkVIF(data,featuresName):
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        VIF_list = [variance_inflation_factor(data,i) for i in range(data.shape[1])]
        VIF = pd.DataFrame({'feature':featuresName,'VIF':VIF_list})
        max_VIF = max(VIF_list)
        print('MAX VIF = ',max_VIF)
        if max_VIF>=10:
            print('===================================================')
            print('VIF_list = ',VIF_list)
    except:
        print("VIF Calculate Error")
    return max_VIF

def press_statistic(y_true, y_pred, xs):
    """
    Calculation of the `Press Statistics <https://www.otexts.org/1580>`_
    """
    res = y_pred - y_true
    hat = xs.dot(np.linalg.pinv(xs))
    den = (1 - np.diagonal(hat))
    sqr = np.square(res/den)
    return sqr.sum()

def predicted_r2(y_true, y_pred, xs):
    """
    Calculation of the `Predicted R-squared <https://rpubs.com/RatherBit/102428>`_
    """
    press = press_statistic(y_true=y_true,
                            y_pred=y_pred,
                            xs=xs
    )

    sst  = np.square(y_true-y_true.mean()).sum()
    return 1 - press / sst

class customerValue:
    def __init__(self, groupName, customerName, data, dataShape, actual, predict, mean, std, coefList, error):
        self.groupName=groupName
        self.customerName=customerName
        self.data=data
        self.dataShape=dataShape
        self.actual=actual
        self.predict=predict
        self.mean=mean
        self.std=std
        self.coefList=coefList
        self.error=error

if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")    