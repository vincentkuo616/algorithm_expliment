"""
Created on Wed Oct  9 17:23:19 2019

@author: vincentkuo
"""

from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import time 
from sklearn.externals import joblib
from sklearn import metrics   ###?算roc和auc
from sklearn.metrics import confusion_matrix
from datetime import datetime, timedelta
from collections import Counter
from catboost import CatBoostClassifier
from catboost import Pool
from termcolor import colored
import matplotlib.pyplot as plt 
import shap

# 52特徵 目前最好參數 67.6768%
# 54特徵 目前最好參數 67.8392% 0.054, 9, 1200, 1


'''
p_learning_rate_Min=0.05
p_depth_Min=7 #:树的深度
p_n_estimators_Min=1300   #: 解决ml问题的树的最大数量
p_l2_leaf_reg_Min=1   # L2正则化系数 default3
p_one_hot_max_size=2  #: 对于某些变量进行one-hot编码
p_loss_function='CrossEntropy' #RMSE  Logloss  MAE  CrossEntropy
p_custom_metric='F1'  #RMSE  Logloss  MAE  CrossEntropy  Recall  Precision  F1  Accuracy  AUC  R2  训练过程中输出的度量值。这些功能未经优化，仅出于信息目的显示。默认None。
p_eval_metric='F1'  #RMSE  Logloss  MAE  CrossEntropy  Recall  Precision  F1  Accuracy  AUC  R2 用于过拟合检验（设置True）和最佳模型选择（设置True）的loss function，用于优化。
p_nan_mode=None #：处理NAN的方法 ，包括Forbidden(禁止存在缺失)，Min(用最小值补)，Max(用最大值补)。默认Min。
p_leaf_estimation_method='Gradient'#：迭代求解的方法，梯度和牛顿  Gradient  Newton
'''



 #loop setting
#========================================================
p_learning_rate_Min=0.054
p_learning_rate_Max=0.06
p_learning_rate_Step=0.01

p_depth_Min=9 #:树的深度
p_depth_Max=12
p_depth_Step=10

p_n_estimators_Min=1300   #: 解决ml问题的树的最大数量
p_n_estimators_Max=1306
p_n_estimators_Step=100

p_l2_leaf_reg_Min=1   # L2正则化系数 default3
p_l2_leaf_reg_Max=3
p_l2_leaf_reg_Step=10


#========================================================
    
    
save = 0 #存檔開關

def main():
    test_subject="test" #僅供LOG檔名使用
    test_dev="local" #測試機器，用以比較不同機台同樣的設定跑速是否有差

    prjRootFolder="C://Users//vincentkuo//Documents//"
    TrainSet = pd.read_csv(prjRootFolder+"vincent//Y191008//UTF8//TrainData_SM_ENN_Y191008.csv",encoding='utf-8')
    TestSet = pd.read_csv(prjRootFolder+"vincent//Y191008//UTF8//TestData_Y191008.csv",encoding='utf-8')  
    allDataSet = pd.read_csv(prjRootFolder+"vincent//Y191008//UTF8//AllData_Ori_Y191008.csv",encoding='utf-8')
    
    reportPath=prjRootFolder+"Report//CATBoost_Y191009.txt"
    
    if save == 1:
        titleStr="test_dev,index,costTime,featureSet,n_estimators,learning_rate,random_state,TN,FP,FN,TP,Precision,Recall,Specificity,Sensitivity,Accuracy,F1-Score,fileName,TimeStamp"
        with open(reportPath, "a") as myfile:
            myfile.write(titleStr+"\n")
        
    index=0
  
    #loopStart
    
    logRoof()
    print("   [ test Loop ]") 
    logFloor()
    
    bM=bestModel("", "", 0, 0, {}, {}, CatBoostClassifier(), {}, 0)

    for featureSet in range (1,2):
        if featureSet==1 :
            endat=61
            X_train = TrainSet.iloc[:,3:endat].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,3:endat].values  
            y_test = TestSet.iloc[:,2].values
            feature_Desc="IV>=0.2"
            
        fname=TrainSet.iloc[0:0,3:endat].columns.values
        
        
        for p_learning_rate in np.arange (p_learning_rate_Min, p_learning_rate_Max, p_learning_rate_Step):
            for p_depth in np.arange (p_depth_Min, p_depth_Max, p_depth_Step):
                for p_n_estimators in np.arange (p_n_estimators_Min, p_n_estimators_Max, p_n_estimators_Step):
                    for p_l2_leaf_reg in np.arange (p_l2_leaf_reg_Min, p_l2_leaf_reg_Max, p_l2_leaf_reg_Step):
                        index = index + 1
                        print("[{}] {} : {} x {}, {}".format(index,feature_Desc,X_train.shape[0],X_train.shape[1],getTimeNow()))
                        print("    >> n_estimator:{}, learning_rate:{}, depth:{}, L2_leaf_reg:{}".format(p_n_estimators,p_learning_rate,p_depth,p_l2_leaf_reg))
                        params = {}                                
                        plst = params.items()
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
                        resultStr="{},{},{},{},{},{},{},{},{},{},{}".format(test_dev,index,costTimeTrain,featureSet,p_learning_rate,p_depth,p_n_estimators,p_l2_leaf_reg,mS.toString(),timestp,getTimeNow())
                        if save == 1:
                            with open(reportPath, "a") as myfile:
                                myfile.write(resultStr+"\n")
                            
                        if mS.f1s>bM.f1s:
                            bM=bestModel(filePath, timestp, index, mS.f1s, X_test, y_test, model, fname, endat)
                                    
    #playsound(prjRootFolder+'//Code//CNoc.mp3')
    logRoof()
    print(">> Loop End @ "+str(index))
    print(">> Best Sel @ {}".format(bM.index))
    catBoost_testFortheBest(bM.X_test, bM.y_test, bM.filePath, bM.fname)
    #getFinalResultForThisRound(prjRootFolder, bM, allDataSet)
    logFloor()
    
def catBoost_train(X_train,y_train,p_learning_rate, p_depth, p_n_estimators,p_l2_leaf_reg,filePath):#train_data,nu,kernel,gamma
    clf = CatBoostClassifier(
                    learning_rate=p_learning_rate
                    ,depth=p_depth
                    ,n_estimators=p_n_estimators
                    ,l2_leaf_reg=p_l2_leaf_reg
                    ,loss_function='CrossEntropy'
                    ,custom_metric='F1'
                    ,eval_metric='F1'
                    ,nan_mode=None
                    ,leaf_estimation_method='Gradient'
                    ,thread_count=-1
                    ,used_ram_limit=None
                    #,p_one_hot_max_size=2
                    )

    model = clf.fit(X_train, y_train)  
    #print("model score: "+model.score") , logging_level='Silent'
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
    
    model=bM.model
    X_all = allDataSet.iloc[:,5:bM.endat].values
    pk_all = allDataSet.iloc[:,0:3].values
    fname = allDataSet.iloc[0:0,5:bM.endat].columns.values
    
    X_importance=X_all
    
    y_predProb = (bM.model.predict(X_all, prediction_type='Probability'))[:,1]
    y_pred = bM.model.predict(X_all)
    
    
    
    outputDF = pd.DataFrame(pk_all)
    outputDF = outputDF.rename(columns={0:'CUSTOMERMCODE',1:'PROP_ID',2:'UrgeMark'})
    outputDF["y_pred"] = y_pred    
    outputDF["y_predProb"] = y_predProb  
    
    #https://liuyanguu.github.io/post/2019/07/18/visualization-of-shap-for-xgboost/
    #https://shap.readthedocs.io/en/latest/
    #http://sofasofa.io/tutorials/shap_xgboost/
    #https://slundberg.github.io/shap/notebooks/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html
    #https://medium.com/ai-academy-taiwan/explain-your-machine-learning-model-by-shap-part-1-228fb2a57119
    explainer = shap.TreeExplainer(bM.model)
    shap_values = explainer.shap_values(X_importance)
    shapDF=pd.DataFrame(shap_values)
    shapDF.columns = fname
    
    print('base value :',explainer.expected_value)
    
    frames = [outputDF, shapDF]
    result = pd.concat(frames, axis=1)
    
    shap.summary_plot(shap_values, X_importance, feature_names=fname, max_display=3)
    
    if 1==2:
        #嫌圖太大改這邊
        shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=80)
        shap.summary_plot(shap_values, X_importance, plot_type='bar', feature_names=bM.fname, max_display=80)
        #shap.force_plot(explainer.expected_value, shap_values[0,:],  bM.fname)
        
        #fim=pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('feature', ascending=False)    
        fs=pd.Series(bM.model.get_best_score()).sort_values(ascending=False)
        
        is_cat = (X_all != float )
        cat_features_index = np.where(is_cat)[0]
        print(cat_features_index)

    
    if save == 1:
       outputFolder=prjRootFolder+"Output//@best//"
       result.to_csv("{}//{}_Result.csv".format(outputFolder,bM.fileName))
       #svpd.to_csv("{}//{}_ResultSHAP.csv".format(outputFolder,bM.fileName))
       #fs.to_csv("{}//{}_FeatureImportance.txt".format(outputFolder,bM.fileName))
       #graph = xgb.to_graphviz(bM.model, num_trees=1, **{'size': str(10)})
       #graph.render(directory=outputFolder,filename=str(bM.fileName)+'_xgb.dot')
       model.save_model(outputFolder+"catBestY1910",format="cbm",export_parameters=None,pool=None)
       
   #loaded_model = joblib.load('xgb.model')


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
    f1s=round(2*precision*recall / (precision + recall),6)

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
    