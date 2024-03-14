import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import time 
from sklearn import svm
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.externals import joblib
from sklearn import metrics   ###?算roc和auc
from sklearn.metrics import confusion_matrix
from termcolor import colored
import warnings
from datetime import datetime, timedelta
import shap
from catboost import CatBoostClassifier
from catboost import Pool
from collections import Counter
import catboost
warnings.filterwarnings('ignore')
shap.initjs()

# prjRootFolder
#  ├ Code : Python Code
#  ├ Data : Dataset
#  ├ Output : Model
#  ├───@best : 本輪最佳解存在這
#  └ Report : log

save=1
save_Shap=0
#存檔開關，0=不存，1=存

 #loop setting
#========================================================
p_learning_rate_Min=0.1
p_learning_rate_Max=0.4
p_learning_rate_Step=0.1

p_depth_Min=3 #:树的深度
p_depth_Max=6
p_depth_Step=1

p_n_estimators_Min=800   #: 解决ml问题的树的最大数量
p_n_estimators_Max=801
p_n_estimators_Step=100

p_l2_leaf_reg_Min=3   # L2正则化系数 default3
p_l2_leaf_reg_Max=4
p_l2_leaf_reg_Step=1

p_one_hot_max_size=2  #: 对于某些变量进行one-hot编码
p_loss_function='CrossEntropy' #RMSE  Logloss  MAE  CrossEntropy
p_custom_metric='F1'  #RMSE  Logloss  MAE  CrossEntropy  Recall  Precision  F1  Accuracy  AUC  R2  训练过程中输出的度量值。这些功能未经优化，仅出于信息目的显示。默认None。
p_eval_metric='F1'  #RMSE  Logloss  MAE  CrossEntropy  Recall  Precision  F1  Accuracy  AUC  R2 用于过拟合检验（设置True）和最佳模型选择（设置True）的loss function，用于优化。
p_nan_mode=None #：处理NAN的方法 ，包括Forbidden(禁止存在缺失)，Min(用最小值补)，Max(用最大值补)。默认Min。
p_leaf_estimation_method='Gradient'#：迭代求解的方法，梯度和牛顿  Gradient  Newton
btest=None
#========================================================

def main():
    log_name="VI_Y191223_B0_8%" #僅供LOG檔名使用

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
    TrainSet_B0 = pd.read_csv(prjRootFolder+"B0_Train_Data_Resampling_SE_8%.csv",encoding='utf-8')
    #TrainSet_B1 = pd.read_csv(prjRootFolder+"B0_Train_Data_Resampling_SE_8%",encoding='utf-8')
    #TestSet_A0 = TestSet.groupby("GROUPA").get_group("A0")
    #TestSet_A1 = TestSet.groupby("GROUPA").get_group("A1")
    TestSet_B0 = TestSet.groupby("GROUPB").get_group("B0")
    #TestSet_B1 = TestSet.groupby("GROUPB").get_group("B1")
    
    A0=[9,10,11,12,13,14,15,16,25,27,28,31,33,35,38,39,45,47,52,56,58,63,67,76,79,82,84,86,87,89,91,92,93,94,95,97,98,99,100,102,103,104,105,106,108,109,111,112,115,123,130,131,142,144,147,150,153,156,157,159,160,163,164]
    A1=[9,10,13,14,20,23,24,27,28,30,33,35,36,39,42,43,48,51,55,56,59,62,64,65,69,78,79,81,83,84,85,86,89,91,92,93,94,95,97,98,99,100,101,102,103,104,105,107,111,113,116,119,121,122,126,128,133,135,136,143,144,146,149,151,152,153,164]
    B0=[9,10,11,13,14,16,19,20,21,25,26,29,30,32,33,36,37,40,44,49,51,53,57,60,62,63,65,66,69,72,73,75,80,83,84,85,86,87,88,90,93,94,96,97,98,99,100,102,103,104,105,107,110,111,112,116,117,118,120,122,127,129,131,132,135,136,137,139,140,141,144,146,148,152,153,154,156,158,159,162,163,164]
    B1=[8,10,11,12,13,14,15,17,18,19,20,22,25,27,28,36,37,38,39,41,46,50,51,54,56,58,61,63,68,70,71,74,77,80,84,85,86,90,91,92,93,94,96,97,98,99,100,103,105,106,108,111,112,114,116,121,125,129,131,134,138,142,145,146,148,152,153,156,157,159,163]
    feature_Y = 2 #Y是第幾個欄位
    sel_f = A1 #選定要用的特徵群
    Train_Data = TrainSet_A1 #要訓練的資料
    Test_Data  = TestSet_A1 #要測試的資料
    
    #存檔開關
    
    if save == 1:
        titleStr="trainSetName,index,costTime,featureSet,p_learning_rate,p_depth,p_n_estimators,p_l2_leaf_reg,TN,FP,FN,TP,Precision,Recall,Specificity,Sensitivity,Accuracy,F1-Score,fileName,TimeStamp"
        with open(reportPath, "a") as myfile:
            myfile.write(titleStr+"\n")
        
    '''
    https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
    '''
    index=0
    
    #loopStart
    logRoof()
    print("   [ test Loop ]") 
    logFloor()
    
    bM=bestModel("", "", 0, 0, {}, {}, CatBoostClassifier(), {}, 0)

    for featureSet in range (1,3):
        endat=sel_f #總特徵個數
        if featureSet==1 :
            feature_Desc="xgboost_B0_Local_SE8%"
            trainSetName="xgboost_B0_Local_SE8%" #文件的分類
            X_train = TrainData.iloc[:,endat].values
            y_train = TrainData.iloc[:,feature_Y].values
            y_train = transformYNto10(y_train)  #檢查 Y 值
            X_test = TestData.iloc[:,endat].values
            y_test = TestData.iloc[:,feature_Y].values
            y_test = transformYNto10(y_test)    #檢查 Y 值
            
        #'''
        elif featureSet==2 :
            TrainData = pd.read_csv(prjRootFolder+"B0_Train_Data_Resampling_AS_8%.csv",encoding='utf-8')
            feature_Desc="xgboost_B0_Local_AS8%"
            trainSetName="xgboost_B0_Local_AS8%" #文件的分類
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
        
        for p_learning_rate in np.arange (p_learning_rate_Min, p_learning_rate_Max, p_learning_rate_Step): 
            for p_depth in np.arange (p_depth_Min, p_depth_Max, p_depth_Step):
                for p_n_estimators in np.arange (p_n_estimators_Min, p_n_estimators_Max, p_n_estimators_Step):
                    for p_l2_leaf_reg in np.arange (p_l2_leaf_reg_Min, p_l2_leaf_reg_Max, p_l2_leaf_reg_Step):
                        index = index + 1
                        print("[{}] {} : {} x {}, {}".format(index,feature_Desc,X_train.shape[0],X_train.shape[1],getTimeNow()))
                        print("    >> n_estimator:{}, learning_rate:{}, depth:{}, L2_leaf_reg:{}".format(p_n_estimators,p_learning_rate,p_depth,p_l2_leaf_reg))
                        timestp = getDatetimeStr()
                        filePath = prjRootFolder+"Output//"+timestp+".pkl"
                        lstime = time.time()
                        #TRAIN
                        model = catBoost_train(X_train,y_train,p_learning_rate, p_depth, p_n_estimators,p_l2_leaf_reg,filePath)
                        #TEST
                        mS = catBoost_testForLoop(X_test,y_test,filePath,fname)
                        
                        letime=time.time()
                        costTimeTrain=round(letime-lstime,2)
                        print("  >> cost: "+str(costTimeTrain)+"s")
                        resultStr="{},{},{},{},{},{},{},{},{},{},{}".format(trainSetName,index,costTimeTrain,featureSet,p_learning_rate,p_depth,p_n_estimators,p_l2_leaf_reg,mS.toString(),timestp,getTimeNow())
                        if save == 1:
                            with open(reportPath, "a") as myfile:
                                myfile.write(resultStr+"\n")
                        if mS.f1s>bM.f1s:
                            bM=bestModel(filePath, timestp, index, mS.f1s, X_test, y_test, model, fname, endat)
                                  
    #playsound(prjRootFolder+'//Code//CNoc.mp3')
    logRoof()
    if bM.index!=0:
        print(">> Loop End @ "+str(index))
        print(">> Best Sel @ {}".format(bM.index))
        catBoost_testFortheBest(bM.X_test, bM.y_test, bM.filePath, bM.fname)
        getFinalResultForThisRound(prjRootFolder, bM, allDataSet)
    else:
        print("No Best Sel @")
    logFloor()

def catBoost_train(X_train,y_train,p_learning_rate, p_depth, p_n_estimators,p_l2_leaf_reg,filePath):#train_data,nu,kernel,gamma
    model = CatBoostClassifier(
                    learning_rate=p_learning_rate, 
                    depth=p_depth,
                    n_estimators=p_n_estimators,
                    l2_leaf_reg=p_l2_leaf_reg,
                    loss_function=p_loss_function,
                    custom_metric=p_custom_metric,
                    eval_metric=p_eval_metric,
                    nan_mode=p_nan_mode,
                    leaf_estimation_method=p_leaf_estimation_method
                    )
    #這邊會跌帶
    model = model.fit(X_train, y_train, logging_level='Silent')    

    with open(filePath, 'wb') as model_file:
        joblib.dump(model, model_file)
    return model
def catBoost_testForLoop(X_test,y_test,filePath,fname):
    with open(filePath, 'rb') as f:
        model = joblib.load(f)
        predictY = model.predict(X_test)
        cm = confusion_matrix(y_test, predictY)
        mS = getCMresults(cm)
        print(colored('  The Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))
    return mS


def catBoost_testFortheBest(X_test,y_test,filePath,fname):
    with open(filePath, 'rb') as f:
        model = joblib.load(f)
        predictY = model.predict(X_test)
        cm = confusion_matrix(y_test, predictY)
        mS=getCMresults(cm)
        print(colored('  The Best Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))        
        #plot_importance(model, max_num_features=100)
        #plt.show()
        
    return mS
    

def getFinalResultForThisRound(prjRootFolder, bM, allDataSet):
    outputFolder=prjRootFolder+"@best//"
    X_importance=bM.X_test
    X_all = allDataSet.iloc[:,bM.endat].values
    pk_all = allDataSet.iloc[:,0:8].values
    #dtest = xgb.DMatrix(X_all, feature_names=bM.fname)
    ans = bM.model.predict(X_all)
    test = np.append(pk_all,np.array([ans]).transpose(),axis=1)
    resultColumnsName = np.append(allDataSet.columns.values[0:8],'Probability')
    result = pd.DataFrame(test,columns=resultColumnsName)
    
    
    # CATBOOST 的 FEATURE SCORE
    X=allDataSet.iloc[:,bM.endat]
    y=allDataSet.iloc[:,2]
    y.replace('Y',1,inplace=True)
    y.replace('N',0,inplace=True)
    feature_score = pd.DataFrame(list(zip(X, bM.model.get_feature_importance(Pool(X, label=y)))), columns=['Feature','Score'])
    fs = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
    print(fs)
    fs = feature_score.sort_values(by='Score', ascending=True, inplace=False, kind='quicksort', na_position='last')
    plt.barh(fs.iloc[:,0].values,fs.iloc[:,1].values,label='Catboost Feature Importance')
    plt.show()
    
    #https://liuyanguu.github.io/post/2019/07/18/visualization-of-shap-for-xgboost/
    #https://shap.readthedocs.io/en/latest/
    #http://sofasofa.io/tutorials/shap_xgboost/
    #https://slundberg.github.io/shap/notebooks/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html
    #https://medium.com/ai-academy-taiwan/explain-your-machine-learning-model-by-shap-part-1-228fb2a57119
    explainer = shap.TreeExplainer(bM.model)
    shap_values = explainer.shap_values(X_importance)
    
    #嫌圖太大改這邊
    shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=len(bM.endat))
    shap.summary_plot(shap_values, X_importance, plot_type='bar', feature_names=bM.fname, max_display=len(bM.endat))
    #shap.force_plot(explainer.expected_value, shap_values[0,:],  bM.fname)
    
    #fim=pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('feature', ascending=False)    
    #fs=pd.Series(bM.model.get_fscore()).sort_values(ascending=False)
    #xgb.plot_tree(bM.model)
    #plt.show()
       
    if save_Shap == 1:
       svpd = pd.DataFrame(shap_values)
       result.to_csv("{}//{}_Result.csv".format(outputFolder,bM.fileName))
       svpd.to_csv("{}//{}_ResultSHAP.csv".format(outputFolder,bM.fileName))
       fi = pd.DataFrame(fs)
       fi.to_csv("{}//{}_Feature_Importance.csv".format(outputFolder,bM.fileName))

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
    def __init__(self, filePath, fileName, index, f1s, Xt, yt, model, fname, endat):
        self.filePath=filePath
        self.index=index
        self.f1s=f1s
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
    
'''
                                             ,,µ▄▄▄▌████▄µww,,                                      
                                      ,wµ▄▄██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▌%mw                                   
                                  µ▄█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▌▄w,                                 
                             wg▄█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓█▌▄µw,                          
                         ,▄█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██▌▄w                    
                      µ▄█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▄w                
                   µ▄█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▀▀▀╜╙²````````╙²╙╜╜▀▀▀▀▓▒▒▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▄              
                 ╟█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀▀²                                `▀▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▌            
               ▄█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                       ╙▀▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▄          
             g█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▀`                                      ,µ▄▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▌         
            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀Ñ`                                        ▄▓▒▒▒▒▌╙▀▒▒▒▒▒▒▒▒▒▒▒▌        
           ╓█▒▒▒▒▒▒▒▒▒▒▒▒▓▀`                                         ╓█▒▒▒▒▒▓   ╙▓▒▒▒▒▒▒▒▒▒▒█       
          @▒▒▒▒▒▒▒▒▒▒▒▒▀╨                                           ╓▒▒▒▒▒▒▒`    ▒▒▒▒▒▒▒▒▒▒▒▒▌      
        ▒▒▒▒▒▒▒▒▒▒▒▒▒▀╜     µ██▌.                                  ,▒▒▒▒▒▒▒Ñ      ╙▒▒▒▒▒▒▒▒▒▒▒▌     
       █▒▒▒▒▒▒▒▒▒▒▒▀`     ▒▒▒▒▒▒▓                                ,g█▒▒▒▒▒▒▒        ▐▒▒▒▒▒▒▒▒▒▒▒▌    
      ╟▒▒▒▒▒▒▒▒▒▒▒▒       ▒▒▒▒▒▒▌                               ╓▒▒▒▒▒▒▒▒▌          ▒▒▒▒▒▒▒▒▒▒▒▒▌   
     ╟▒▒▒▒▒▒▒▒▒▒▒▒▌        ▒▒▒▒▒▌               ,,              ▒▒▒▒▒▒▒▒▓`          ▒▒▒▒▒▒▒▒▒▒▒▒▒L  
     ▐▒▒▒▒▒▒▒▒▒▒▒▒▌        ▐▒▒▒▒▒Γ           ╓▄▄▒▒U             ▒▒▒▒▒▒▒▒▌           ▐▒▒▒▒▒▒▒▒▒▒▒▒Ñ  
     ▐▒▒▒▒▒▒▒▒▒▒▒▓`         ▒▒▒▒▒[          ¡█▒▒▒▒▒█            ▐▒▒▒▒▒▒▒            ▐▒▒▒▒▒▒▒▒▒▒▒▒   
     ▐▒▒▒▒▒▒▒▒▒▒▒▌          ▒▒▒▒▒L          ▒▒▒▒▒▒▒▒▌          ╓█▒▒▒▒▒▒▒            ]▒▒▒▒▒▒▒▒▒▒▒▒▌  
   ▐▌▒▒▒▒▒▒▒▒▒▒▒▒▌          ]▓▒▒▒H         ▒▒▒▒▒▒▒▒▒▒        ,█▒▒▒▒▒▒▒▒▌            ]▒▓▒▒▒▒▒▒▒▒▒▒▌  
  ╓▒▒▒▒▒▒▒▒▒▒▒▒▒▒█w          ▒▒▒▒▌        ,▓▒▒▒▒▒▒▒▒▒▌       ╠▒▒▒▒▒▒▒▒▒             ]▓░▒▒▒▒▒▒▒▒▒Ñ   
  ╟▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▌          ╢▒▒▒▌        ▒▒▒▓▓▒▒▒▒▒▒▒▌      ▒▒▒▒▒▒▒▒▒▒              ` ▒▒▒▒▒▒▒▒▒▌   
  ╟▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒w           ▒▒▒▒       ▒▒▒▌  ▒▒▒▒▒▒▒▒▌     ▒▒▒▒▒▒▒▒▒H               j▓▒▒▒▒▒▒▒▒▒   
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▌           ▒▒▒▒      @▒▒▓`  ▒▒▒▒▒▒▒▒▒▌   ▐▒▒▒▒▒▒▒▒▌                ▐▒▒▒▒▒▒▒▒▒▒   
   ▄▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▌          ▒▒▒▒     ▒▒▒▒M    ▓▒▒▒▒▒▒▒▒▌ d▒▒▒▒▒▒▒▒▓                 ▒▒▒▒▒▒▒▒▒▒▒   
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ╟▒▒▒∩   ▒▒▒▒Ñ      ▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▓`                ▐▒▒▒▒▒▒▒▒▒▒▌   
    ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▌           ▒▒▒   ▒▒▒▒▒       ╙▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓`                 ▓▒▒▒▒▒▒▒▒▒▒    
     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒█p          ▒▒▒ ,█▒▒▒▌         ╙▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                 d█▒▒▒▒▒▒▒▒▒     
      ╙▒▒▒▒▒▒▒▒▒▒▒▒▒▒L        j▒▒▒▄▓▒▒▒▒            ▒▒▒▒▒▒▒▒▒▒▒▒▒Ñ                ,▒▒▒▒▒▒▒▒▒▓╨      
       ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒,        ▒▒▒▒▒▒▒╨              ╙▒▒▒▒▒▒▒▒▒▌`                d▓▒▒▒▒▒▒▒▒▌        
        ▐▒▒▒▒▒▒▒▒▒▒▒▒▒▓▌µ      ▒▒▒▒▒Ñ                  ╙▓▒▒▒▒▀Ñ                ,█▒▒▒▒▒▒▒▓╨          
         `▀▀▒▒▒▒▒▒▒▒▒▒▒▒▒█▄,   ▒▒▒▓╜.                    `"`                 ,▄▒▒▒▒▒▓▀▀▒`           
             ▀▓▒▒▒▒▒▒▒▒▒▒▒▒▒█▄N▒▀╨Ñµ▐▌ ▒,╙y , ,                              ▒▒▒▒▀▀▌`               
               Ñ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▌▄▒▒▒█▒▒▌▒▌▒▌▒w▌╓                         ▄█▒▀▀`,M                 
                 Ñ▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▌█H                      ╓█▒▌`                      
                   ╙▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▌                    µ█▓▀╜                        
                      ▀▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▌Ñ╜`            ,µ@▀╨`                           
                        ]▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓N           ,µM╨`                                
                       ╙▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▌         ^`                                      
                         `▀▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▌µ                                               
                            ╙▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▌                                                 
                               `ⁿ▒▓▓▓▒▒▒▒▒▒▒▒▒▓╝  ╙`                                                
                                ╙` ╓▒▀▓▓▀▀▀╨`▒w                                                     
                                  ╙`   ╙      `                           
'''    

