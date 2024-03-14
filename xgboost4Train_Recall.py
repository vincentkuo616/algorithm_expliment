# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:07:45 2019

Updated on Thu Dec 19 11:57:45 2019

@author: vincentkuo
"""

import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import time 
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import metrics   ###?算roc和auc
from sklearn.metrics import confusion_matrix
from termcolor import colored
import warnings
from datetime import datetime, timedelta
import shap
from math import ceil
warnings.filterwarnings('ignore')
shap.initjs()

# prjRootFolder
#  ├ Code : Python Code
#  ├ Data : Dataset
#  ├ Output : Model
#  ├───@best : 本輪最佳解存在這
#  └ Report : log

save=1
save_ShapDoc=0  #Shap 值文字檔
save_ShapPlot=0 #Shap 值圖片檔
#存檔開關，0=不存，1=存

A0=[9,10,11,12,13,14,15,16,25,27,28,31,33,35,38,39,45,47,52,56,58,63,67,76,79,82,84,86,87,89,91,92,93,94,95,97,98,99,100,102,103,104,105,106,108,109,111,112,115,123,130,131,142,144,147,150,153,156,157,159,160,163,164]
A1=[9,10,13,14,20,23,24,27,28,30,33,35,36,39,42,43,48,51,55,56,59,62,64,65,69,78,79,81,83,84,85,86,89,91,92,93,94,95,97,98,99,100,101,102,103,104,105,107,111,113,116,119,121,122,126,128,133,135,136,143,144,146,149,151,152,153,164]
B0=[9,10,11,13,14,16,19,20,21,25,26,29,30,32,33,36,37,40,44,49,51,53,57,60,62,63,65,66,69,72,73,75,80,83,84,85,86,87,88,90,93,94,96,97,98,99,100,102,103,104,105,107,110,111,112,116,117,118,120,122,127,129,131,132,135,136,137,139,140,141,144,146,148,152,153,154,156,158,159,162,163,164]
B1=[8,10,11,12,13,14,15,17,18,19,20,22,25,27,28,36,37,38,39,41,46,50,51,54,56,58,61,63,68,70,71,74,77,80,84,85,86,90,91,92,93,94,96,97,98,99,100,103,105,106,108,111,112,114,116,121,125,129,131,134,138,142,145,146,148,152,153,156,157,159,163]
sel_f = B0 # 挑選之特徵集
feature_Y=2 #Y是第幾個欄位
deleteList = [10,154,136,139,137,60,132,140,118,75,110,40,104,135,117,129]

def main():
    txt_name="VI_Y191220_B0" #僅供LOG檔名使用    
    
    #資料結構請參考上面
    prjRootFolder="C://Users//vincentkuo//Documents//vincent_2//"
    #TrainSet = pd.read_csv(prjRootFolder+"Train_Data_rename.csv",encoding='utf-8')
    #TestSet = pd.read_csv(prjRootFolder+"Test_Data_rename.csv",encoding='utf-8')
    reportPath=prjRootFolder+"Report//XGBoost_"+txt_name+"_log"+getDatetimeStr()+".txt"
    
    allDataSet = pd.read_csv(prjRootFolder+"ALL_DATA_rename.csv",encoding='utf-8')
    TrainSet = allDataSet.groupby("SAMPLE").get_group("TRAIN")
    TestSet = allDataSet.groupby("SAMPLE").get_group("TEST")
    #allDataSet = TrainSet.append(TestSet)
    
    #將資料分組
    #TrainSet_A0 = pd.read_csv(prjRootFolder+"ALL_DATA_rename.csv",encoding='utf-8')
    #TrainSet_A1 = TrainSet.groupby("GROUPA").get_group("A1")
    TrainSet_B0 = pd.read_csv(prjRootFolder+"B0_Train_Data_Resampling_SE_13%.csv",encoding='utf-8')
    #TrainSet_B1 = pd.read_csv(prjRootFolder+"ALL_DATA_rename.csv",encoding='utf-8')
    #TestSet_A0 = TestSet.groupby("GROUPA").get_group("A0")
    #TestSet_A1 = TestSet.groupby("GROUPA").get_group("A1")
    TestSet_B0 = TestSet.groupby("GROUPB").get_group("B0")
    #TestSet_B1 = TestSet.groupby("GROUPB").get_group("B1")
    
    TrainData = TrainSet_B0 # 要跑的訓練集
    TestData  = TestSet_B0  # 要跑的測試集

    #存檔開關
    if save == 1:
        titleStr="trainSetName,index,costTime,featureSet,max_depth,gamma,subsample,scale_pos_weight,eta,min_child_weight,estimators,TN,FP,FN,TP,Precision,Recall,Specificity,Sensitivity,Accuracy,F1-Score,fileName,TimeStamp"
        with open(reportPath, "a") as myfile:
            myfile.write(titleStr+"\n")
        
    '''
    https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
    '''
    index=0
    
    #loop setting
    #========================================================
    #最大深度 建議: [3-10]
    depthMin=3
    depthMax=12
    depthStep=2
    #分裂後損失函數大於此閾值才會長子結點，增加:避免overfitting 建議: [0-0.2]
    gammaMin=0.1
    gammaMax=0.3
    gammaStep=0.1
    #對於每棵樹隨機採樣的比例，降低:避免overfitting；過低:underfitting 建議: [0.5-0.9]
    subsampleMin=0.7
    subsampleMax=0.91
    subsampleStep=200
    #colsample_bytree 控制每顆樹隨機採樣的列數的佔比 建議: [0.5-0.9]
    cbMin=0.7
    cbMax=0.91
    cbStep=200
    #learning rate
    etaMin=0.1
    etaMax=0.6
    etaStep=0.2
    #min_child_weight 決定最小葉子節點樣本權重和，加權和低於這個值時，就不再分裂產生新的葉子節點 建議: [1]
    mcwMin=2
    mcwMax=3
    mcwStep=100
    #boost迭代次數
    itersMin=1000
    itersMax=1300
    itersStep=100
    #========================================================
    
    loopNum = ceil((depthMax-depthMin)/depthStep)*ceil((gammaMax-gammaMin)/gammaStep)*ceil((subsampleMax-subsampleMin)/subsampleStep)\
    *ceil((cbMax-cbMin)/cbStep)*ceil((etaMax-etaMin)/etaStep)*ceil((mcwMax-mcwMin)/mcwStep)*ceil((itersMax-itersMin)/itersStep)
    print(loopNum)
    
    #loopStart
    
    logRoof()
    print("   [ test Loop ]") 
    logFloor()
    
    bM=bestModel("", "", 0, 0, 0, {}, {}, xgb.core.Booster(), {}, 0)

    for featureSet in range(1,2):
        featureSet = 1
        endat=sel_f #總特徵個數
        if featureSet==1 :
            feature_Desc="xgboost_B0_Local"
            trainSetName="xgboost_B0_Local" #文件的分類
            X_train = TrainData.iloc[:,endat].values
            y_train = TrainData.iloc[:,feature_Y].values
            y_train = transformYNto10(y_train)  #檢查 Y 值
            X_test = TestData.iloc[:,endat].values
            y_test = TestData.iloc[:,feature_Y].values
            y_test = transformYNto10(y_test)    #檢查 Y 值
            
        #'''
        elif featureSet==2 :
            endat=16
            feature_Desc="IV>=0.2+HandMade"
            trainSetName="xgboost_test_2" #文件的分類
            X_train = TrainData.iloc[:,endat].values
            y_train = TrainData.iloc[:,feature_Y].values
            y_train = transformYNto10(y_train)  #檢查 Y 值
            X_test = TestData.iloc[:,endat].values
            y_test = TestData.iloc[:,feature_Y].values
            y_test = transformYNto10(y_test)    #檢查 Y 值
        
        else :
            endat=21
            feature_Desc="All+OneHot"
            trainSetName="xgboost_test_3" #文件的分類
            X_train = TrainData.iloc[:,endat].values
            y_train = TrainData.iloc[:,feature_Y].values
            y_train = transformYNto10(y_train)  #檢查 Y 值
            X_test = TestData.iloc[:,endat].values
            y_test = TestData.iloc[:,feature_Y].values
            y_test = transformYNto10(y_test)    #檢查 Y 值
        #'''
        fname=TrainSet.iloc[0:0,endat].columns.values
        
        for p_depth in np.arange (depthMin, depthMax, depthStep): #max_depth
            for p_gamma in np.arange(gammaMin, gammaMax, gammaStep): #gamma
                for p_ss in np.arange(subsampleMin, subsampleMax, subsampleStep):
                    for p_cb in np.arange(cbMin, cbMax, cbStep):
                        for p_eta in np.arange (etaMin, etaMax, etaStep): 
                            for p_mcw in np.arange (mcwMin, mcwMax, mcwStep):
                                for iters in np.arange (itersMin, itersMax, itersStep):
                                    index += 1
                                    print("[{}] {} : {} x {}, {}".format(index,feature_Desc,X_train.shape[0],X_train.shape[1],getTimeNow()))
                                    print("    >> max_depth:{}, gamma:{}, ss:{}, cb:{}, eta:{}, mcw:{}, estimators:{}".format(p_depth,p_gamma,p_ss,p_cb,p_eta,p_mcw,iters))
                                    params = {
                                        'booster': 'gbtree',
                                        'objective': 'binary:logistic',
                                        'gamma': p_gamma,
                                        'max_depth': p_depth,
                                        'lambda': 2,
                                        'subsample': p_ss,
                                        'colsample_bytree': p_cb, 
                                        'min_child_weight': p_mcw, 
                                        'silent': 1,
                                        'eta': p_eta, 
                                        'nthread': -1,
                                        'eval_metric' : 'error',
                                        'scale_pos_weight': 1, #如果出現嚴重的不平衡，則應使用大於0的值，因為它有助於加快收斂速度。 負樣本個數/正樣本個數 
                                        #上面這條有的人說可以調，有的人說調了沒用，所以請自行參考斟酌
                                    }
                                                    
                                    plst = params.items()
                                    dtrain = xgb.DMatrix(X_train, y_train, feature_names=fname)
                                    timestp = getDatetimeStr()
                                    filePath = prjRootFolder+"Output//"+timestp+".pkl"
                                    lstime = time.time()
                                    
                                    #TRAIN
                                    model = xgBoost_train(plst, dtrain, iters, filePath)
                                    #TEST
                                    mS = xgBoost_testForLoop(X_test,y_test,filePath,fname)
                                    
                                    letime=time.time()
                                    costTimeTrain=round(letime-lstime,2)
                                    print("  >> cost: "+str(costTimeTrain)+"s")
                                    resultStr="{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(trainSetName,index,costTimeTrain,featureSet,p_depth,p_gamma,p_ss,p_cb,p_eta,p_mcw,iters,mS.toString(),timestp,getTimeNow())
                                    if save == 1:
                                        with open(reportPath, "a") as myfile:
                                            myfile.write(resultStr+"\n")
                                        
                                    if mS.recall>bM.recall:
                                        bM=bestModel(filePath, timestp, index, mS.f1s, mS.recall, X_test, y_test, model, fname, endat)
        #sel_f.remove(i)
                                    
    logRoof()
    if bM.index!=0:
        print(">> Loop End @ "+str(index))
        print(">> Best Sel @ {}".format(bM.index))
        xgBoost_testFortheBest(bM.X_test, bM.y_test, bM.filePath, bM.fname, prjRootFolder)
        getFinalResultForThisRound(prjRootFolder, bM, allDataSet)
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
        ans = model.predict(dtest)
        cm = confusion_matrix(y_test, (ans>0.5))
        mS = getCMresults(cm)
        print(colored('  The Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> Recall = ', 'blue'), colored(toPercentage(mS.recall), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))
    return mS

def xgBoost_testFortheBest(X_test,y_test,filePath,fname,prjRootFolder):
    with open(filePath, 'rb') as f:
        model = joblib.load(f)
        # pred the test data
        dtest = xgb.DMatrix(X_test, feature_names=fname)
        ans = model.predict(dtest)
        cm = confusion_matrix(y_test, (ans>0.5))
        mS=getCMresults(cm)
        print(colored('  The Best Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> Recall = ', 'blue'), colored(toPercentage(mS.recall), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))
        plot_importance(model, max_num_features=20)
        if save_ShapPlot == 1:
            plt.savefig(prjRootFolder+"@best//XGB_ImportancePlot.png",bbox_inches='tight')
        plt.show()
        
    return mS
    

def getFinalResultForThisRound(prjRootFolder, bM, allDataSet):
    outputFolder=prjRootFolder+"@best//"
    X_importance=bM.X_test
    X_all = allDataSet.iloc[:,bM.endat].values
    pk_all = allDataSet.iloc[:,0:8].values
    dtest = xgb.DMatrix(X_all, feature_names=bM.fname)
    ans = bM.model.predict(dtest)
    test = np.append(pk_all,np.array([ans]).transpose(),axis=1)
    resultColumnsName = np.append(allDataSet.columns.values[0:8],'Probability')
    result = pd.DataFrame(test,columns=resultColumnsName)
    #https://liuyanguu.github.io/post/2019/07/18/visualization-of-shap-for-xgboost/
    #https://shap.readthedocs.io/en/latest/
    #http://sofasofa.io/tutorials/shap_xgboost/
    #https://slundberg.github.io/shap/notebooks/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html
    #https://medium.com/ai-academy-taiwan/explain-your-machine-learning-model-by-shap-part-1-228fb2a57119
    explainer = shap.TreeExplainer(bM.model)
    shap_values = explainer.shap_values(X_importance)
    
    #嫌圖太大改這邊
    if save_ShapPlot == 1:
        shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=len(bM.endat), show=False)
        plt.savefig(outputFolder+bM.fileName+"_shap_summary.png",bbox_inches='tight')
        #這邊存檔會有圖形疊加的問題
        #shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, plot_type='bar', max_display=len(bM.endat), show=False)
        #plt.savefig(outputFolder+bM.fileName+"_shap_summary_bar.png",bbox_inches='tight')
        graph = xgb.to_graphviz(bM.model, num_trees=1, **{'size': str(10)})
        graph.render(directory=outputFolder,filename=str(bM.fileName)+'_xgb.dot')
        #bM.model.save_model('{}_xgb.model'.format(bM.fileName))
    else:
        shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=len(bM.endat), show=True)
        shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, plot_type='bar', max_display=len(bM.endat), show=True)
  
    fs=pd.Series(bM.model.get_fscore()).sort_values(ascending=False)


    if save_ShapDoc == 1:
       svpd = pd.DataFrame(shap_values)
       result.to_csv("{}//{}_Result.csv".format(outputFolder,bM.fileName))
       svpd.to_csv("{}//{}_ResultSHAP.csv".format(outputFolder,bM.fileName))
       fs.to_csv("{}//{}_FeatureImportance.txt".format(outputFolder,bM.fileName))
    

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
    def __init__(self, filePath, fileName, index, f1s, recall, Xt, yt, model, fname, endat):
        self.filePath=filePath
        self.index=index
        self.f1s=f1s
        self.recall=recall
        self.X_test=Xt
        self.y_test=yt
        self.fileName=fileName
        self.model=model
        self.fname=fname
        self.endat=endat

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
    
    

