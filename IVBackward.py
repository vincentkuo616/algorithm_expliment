# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:22:40 2020

@author: vincentkuo
"""

import xgboost as xgb
import time 
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from termcolor import colored
import warnings
from datetime import datetime, timedelta
import shap
import copy
warnings.filterwarnings('ignore')
shap.initjs()

# prjRootFolder
#  ├ Code : Python Code
#  ├ Data : Dataset
#  ├ Output : Model
#  └ Report : log
'''
使用說明書：
1.要調整之變數:
    featureSelection         -->本次要backward的特徵
    f1Threshold      -->預設F1最低門檻，若不知道就打0也可以
    label     -->我們要預測的欄位
    txt_name      -->存檔log檔名
    prjRootFolder -->專案資料夾
    reportPath    -->loop過程紀錄檔名
    allDataSet    -->所有資料
    TrainSet      -->training用
'''

deleteArray=[]
#預設本輪最大值
thisRoundMaxF1 = 0 
#存檔開關，0=不存，1=存
save=1
#預設 F1的門檻
f1Threshold=0
# 挑選之特徵集
featureSelection = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380] 
featureSelection = [11,13,16,18,19,20,21,29,36,44,45,50,52,54,62,70,76,84,85,86,96,104,107,109,111,112,114,126,133,137,146,152,164,178,190,197,199,209,215,225,228,229,234,235,236,243,246,251,263,264,265,267,275,296,299,302,311,320,331,335,341,348,362,363,366,367,373,375,376,12,14,15,17,22]
#Y是第幾個欄位
label=2

def main():
    ori_del_length = len(deleteArray)
    backward()
    print("delete=",deleteArray)
    while(len(featureSelection)>2 and ori_del_length<len(deleteArray)):
        ori_del_length = len(deleteArray)
        backward()
        print("delete=",deleteArray)
    print('-----------------------------------------------')    
    print('-----------------------------------------------')    
    print('-----------------------------------------------')    
    print('The useless Feature List is =',deleteArray)
    print('The Best Feature List is =',featureSelection)

def backward():
    txt_name ="VI_Y200120_74Tuning" #僅供LOG檔名使用    
    
    prjRootFolder="C://Users//vincentkuo//Documents//vincent_TW//"

    allDataSet = pd.read_csv(prjRootFolder+"A1_IV_Y200117.csv",encoding='utf-8')

    #將資料分組
    TrainData = pd.read_csv(prjRootFolder+"A1_IV_Y200117_8%.csv",encoding='utf-8')
    TestData = allDataSet.groupby("SAMPLE").get_group("TEST").groupby("GROUPA").get_group("A1")

    reportPath=prjRootFolder+"Report//XGBoost_"+txt_name+"_log"+getDatetimeStr()+".txt"

    #存檔開關
    if save == 1:
        titleStr="trainSetName,index,costTime,delete_feature,max_depth,gamma,subsample,scale_pos_weight,eta,min_child_weight,estimators,TN,FP,FN,TP,Precision,Recall,Specificity,Sensitivity,Accuracy,F1-Score,fileName,TimeStamp,feature_list"
        with open(reportPath, "a") as myfile:
            myfile.write(titleStr+"\n")
        
    index=0
    
    logRoof()
    print("   [ test Loop ]") 
    logFloor()
    
    bM=bestModel("", "", 0, 0, {}, {}, xgb.core.Booster(), {}, 0,"")
  
    #刪除本輪的的特徵
    for i in  deleteArray:
        try:
            featureSelection.remove(i)
        except:
            print('bomb')
        
    A1_len = len(featureSelection)
        
    for featureSet in range (0,A1_len):
        tmpFea = copy.deepcopy(featureSelection)
        tmpFea.pop(featureSet)#刪除特定元素
        endat = tmpFea
        if 1==1 :
            feature_Desc="A1_backward"
            trainSetName="A1_backward" #文件的分類
            X_train = TrainData.iloc[:,endat].values
            y_train = TrainData.iloc[:,label].values
            y_train = transformYNto10(y_train)  #檢查 Y 值
            X_test = TestData.iloc[:,endat].values
            y_test = TestData.iloc[:,label].values
            y_test = transformYNto10(y_test)    #檢查 Y 值
            
        fname=TestData.iloc[0:0,endat].columns.values
        print('feature_list=',tmpFea)

        index += 1
        print("[{}] {} : {} x {}, {}".format(index,feature_Desc,X_train.shape[0],X_train.shape[1],getTimeNow()))
        print("    >> max_depth:{}, gamma:{}, ss:{}, cb:{}, eta:{}, mcw:{}, estimators:{}".format(6,0,1,1,0.3,1,1800))
        print("feature_Columns="+str(featureSet))
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'max_depth': 6,
            'gamma': 0,
            'subsample': 1,
            'colsample_bytree': 1, 
            'eta': 0.3, 
            'min_child_weight': 1, 
            'lambda': 2,
            'silent': 1,            
            'nthread': -1,
            'eval_metric' : 'error',
            'scale_pos_weight': 1, #如果出現嚴重的不平衡，則應使用大於0的值，因為它有助於加快收斂速度。 負樣本個數/正樣本個數 
            #上面這條有的人說可以調，有的人說調了沒用，所以請自行參考斟酌
        }
                        
        plst = params.items()
        dtrain = xgb.DMatrix(X_train, y_train, feature_names=fname)
        time.sleep(0.8)
        timestp = getDatetimeStr()
        filePath = prjRootFolder+"Output//"+timestp+".pkl"
        lstime = time.time()
        
        #TRAIN
        model = xgBoost_train(plst, dtrain, 1800, filePath)
        #TEST
        mS = xgBoost_testForLoop(X_test,y_test,filePath,fname)
        
        letime=time.time()
        costTimeTrain=round(letime-lstime,2)
        print("  >> cost: "+str(costTimeTrain)+"s")
        #把list存到紀錄檔
        fea_list=""
        for fl in endat:
            fea_list = fea_list+';'+str(fl)
        resultStr="{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(trainSetName,index,costTimeTrain,featureSelection[featureSet],6,0,1,1,0.3,1,1800,mS.toString(),timestp,getTimeNow(),fea_list)
        if save == 1:
            with open(reportPath, "a") as myfile:
                myfile.write(resultStr+"\n")
            
        if mS.f1s>bM.f1s:
            bM=bestModel(filePath, timestp, index, mS.f1s, X_test, y_test, model, fname, endat,featureSelection[featureSet])
            
                                    
    logRoof()
    if bM.index!=0:
        print(">> Loop End @ "+str(index))
        print(">> Best Sel @ {}".format(bM.index))
        xgBoost_testFortheBest(bM.X_test, bM.y_test, bM.filePath, bM.fname, prjRootFolder,bM.featureSet)
        
    else:
        print("No Best Sel @")
    logFloor()

def xgBoost_train(plst, dtrain, num_rounds,filePath):#train_data,nu,kernel,gamma
    model = xgb.train(plst, dtrain, num_rounds)
    with open(filePath, 'wb') as model_file:
        joblib.dump(model, model_file)
    return model

def xgBoost_testForLoop(X_test,y_test,filePath,fname):
    with open(filePath, 'rb') as f:
        model = joblib.load(f)
        # pred the test data
        dtest = xgb.DMatrix(X_test, feature_names=fname)
        #dtest = xgb.DMatrix(X_test)
        ans = model.predict(dtest)
        cm = confusion_matrix(y_test, (ans>0.5))
        mS = getCMresults(cm)
        print(colored('  The Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))
    return mS

def xgBoost_testFortheBest(X_test,y_test,filePath,fname,prjRootFolder,delete_f):
    with open(filePath, 'rb') as f:
        global f1Threshold 
        global thisRoundMaxF1
        
        model = joblib.load(f)
        # pred the test data
        dtest = xgb.DMatrix(X_test, feature_names=fname)
        #dtest = xgb.DMatrix(X_test)
        ans = model.predict(dtest)
        cm = confusion_matrix(y_test, (ans>0.5))
        mS=getCMresults(cm)
        print(colored('  The Best Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))
        
        thisRoundMaxF1 = mS.f1s #設定最大F1

        if(float(thisRoundMaxF1)>=float(f1Threshold)):
            deleteArray.append(delete_f)
            f1Threshold=mS.f1s
        print(' つ◕_◕ ༽つ ༼つ͡◕_ ͡◕ ༽つ ༼ つ◉ㅂ◉ ༽つ ༼ っ◕﹏◕ ༽っ')
        print(' つ◕_◕ ༽つ ༼つ͡◕_ ͡◕ ༽つ ༼ つ◉ㅂ◉ ༽つ ༼ っ◕﹏◕ ༽っ')
        print('------------This Round is Finish -------------')
        print(' つ◕_◕ ༽つ ༼つ͡◕_ ͡◕ ༽つ ༼ つ◉ㅂ◉ ༽つ ༼ っ◕﹏◕ ༽っ')
        print(' つ◕_◕ ༽つ ༼つ͡◕_ ͡◕ ༽つ ༼ つ◉ㅂ◉ ༽つ ༼ っ◕﹏◕ ༽っ')
        
    return mS
    

def esttimeplz(loopCount,sec):
    est=int(loopCount)*sec   #estimate time
    d = datetime.now()+ timedelta(seconds = est)
    return '{0:%Y/%m/%d %H:%M:%S}'.format(d)

def getDatetimeStr():
    d = datetime.now()
    str='{:%Y%m%d%H%M%S}'.format(d)
    return str
 

def getTimeNow():
    return '{0:%Y/%m/%d %H:%M:%S}'.format(datetime.now())

def getCMresults(cm):
    C_matrix = cm
    tn=C_matrix[0, 0]
    fp=C_matrix[0, 1]
    fn=C_matrix[1, 0]
    tp=C_matrix[1, 1]
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision= tp / (tp + fp)
    accuracy=(tp + tn) / (tp + fn + fp + tn)
    recall= tp / (tp+fn)
    f1s=round(2*tp/(2*tp+fn+fp),6)

    return mScore(tn, fp, fn, tp, precision, recall, specificity,sensitivity, accuracy, f1s)
    
def toPercentage(floatNum):
    return "{percent:.4%}".format(percent=floatNum)

def logS(title):
    print("\n ========================================== ")
    print("   [ "+title+" Start ] "+getTimeNow())
    print("\n")    
    global Stime
    Stime=time.time()
    #print(" ===================================== ")

def logE(title):
    Etime = time.time()
    global Stime
    #print("\n ===================================== ")
    print("\n")
    print("   Cost: "+"{}".format(round(Etime-Stime,2))+"s" )
    print("   [ "+title+" End ] "+getTimeNow())
    print(" ========================================== ")

def log(title):
    print("\n ========================================== ")
    print("   [ "+title+" ] "+getTimeNow())
    print(" ========================================== ")
    
def logRoof():
    print("\n ========================================== ")

def logFloor():
    print("   "+getTimeNow())
    print(" ========================================== ")

def transformYNto10(yn):
    try:
        yn[yn=='Y']=1
        yn[yn=='N']=0
        yn = yn.astype(np.int64)
        return yn
    except:
        print("Something Wrong at transformYNto10")
    

class bestModel:
    def __init__(self, filePath, fileName, index, f1s, Xt, yt, model, fname, endat,featureSet):
        self.filePath=filePath
        self.index=index
        self.f1s=f1s
        self.X_test=Xt
        self.y_test=yt
        self.fileName=fileName
        self.model=model
        self.fname=fname
        self.endat=endat
        self.featureSet=featureSet

class mScore:
    def __init__(self, tn, fp, fn, tp, precision, recall, specificity,sensitivity, accuracy, f1s):
        self.tn=tn
        self.fp=fp
        self.fn=fn
        self.tp=tp
        self.precision=precision
        self.recall=recall
        self.specificity=specificity
        self.sensitivity=sensitivity
        self.accuracy=accuracy
        self.f1s=f1s        
    def toString(self):
        return "{},{},{},{},{},{},{},{},{},{}".format(self.tn, self.fp, self.fn, self.tp, self.precision, self.recall, self.specificity,self.sensitivity, self.accuracy, self.f1s)
    
if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")    
    
    

