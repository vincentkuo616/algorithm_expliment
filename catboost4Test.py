# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:07:45 2019

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
from catboost import CatBoostClassifier,Pool
from termcolor import colored
import xgboost as xgb
import shap
from matplotlib import pyplot as plt



#目前最好 這版不能再調整了!!!!



 #loop setting
#========================================================
p_learning_rate_Min=0.05
p_learning_rate_Max=0.06
p_learning_rate_Step=0.05

p_depth_Min=7 #:树的深度
p_depth_Max=8
p_depth_Step=3

p_n_estimators_Min=1300   #: 解决ml问题的树的最大数量
p_n_estimators_Max=1301
p_n_estimators_Step=100

p_l2_leaf_reg_Min=1   # L2正则化系数 default3
p_l2_leaf_reg_Max=2
p_l2_leaf_reg_Step=1

p_one_hot_max_size=2  #: 对于某些变量进行one-hot编码
p_loss_function='CrossEntropy' #RMSE  Logloss  MAE  CrossEntropy
p_custom_metric='F1'  #RMSE  Logloss  MAE  CrossEntropy  Recall  Precision  F1  Accuracy  AUC  R2  训练过程中输出的度量值。这些功能未经优化，仅出于信息目的显示。默认None。
p_eval_metric='F1'  #RMSE  Logloss  MAE  CrossEntropy  Recall  Precision  F1  Accuracy  AUC  R2 用于过拟合检验（设置True）和最佳模型选择（设置True）的loss function，用于优化。
p_nan_mode=None #：处理NAN的方法 ，包括Forbidden(禁止存在缺失)，Min(用最小值补)，Max(用最大值补)。默认Min。
p_leaf_estimation_method='Gradient'#：迭代求解的方法，梯度和牛顿  Gradient  Newton

#========================================================
    
    
save = 0 #存檔開關

def main():
    test_subject="test" #僅供LOG檔名使用
    test_dev="SM_ENN_10%Y" #測試機器，用以比較不同機台同樣的設定跑速是否有差

    #資料結構請參考上面
    prjRootFolder="C://Users//vincentkuo//Documents//vincent//Y191028//"
    TrainSet = pd.read_csv(prjRootFolder+"TrainData_A_Subset.csv",encoding='utf-8')
    TestSet = pd.read_csv(prjRootFolder+"TestData_A_Subset.csv",encoding='utf-8')  
    reportPath=prjRootFolder+"CATBoost_WB_SM_ENN_10%_Y191001.txt"
    
    allDataSet = TrainSet.append(TestSet)
    
    #存檔開關
    if save == 1:
        titleStr="test_dev,index,costTime,featureSet,n_estimators,learning_rate,random_state,TN,FP,FN,TP,Precision,Recall,Specificity,Sensitivity,Accuracy,F1-Score,fileName,TimeStamp"
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

    for featureSet in range (1,2):
        if featureSet==1 :
            endat=57
            X_train = TrainSet.iloc[:,5:57].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,5:57].values  
            y_test = TestSet.iloc[:,2].values
            feature_Desc="IV>=0.2"
        '''    
        elif featureSet==2 :
            endat=65
            feature_Desc="IV>=0.2+HandMade"
            X_train = TrainSet.iloc[:,3:endat].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,3:endat].values  
            y_test = TestSet.iloc[:,2].values
        '''
            
        fname=TrainSet.iloc[0:0,1:endat-2].columns.values
        
        for p_learning_rate in np.arange (p_learning_rate_Min, p_learning_rate_Max, p_learning_rate_Step):
            for p_depth in np.arange (p_depth_Min, p_depth_Max, p_depth_Step):
                for p_n_estimators in np.arange (p_n_estimators_Min, p_n_estimators_Max, p_n_estimators_Step):
                    for p_l2_leaf_reg in np.arange (p_l2_leaf_reg_Min, p_l2_leaf_reg_Max, p_l2_leaf_reg_Step):
                        index = index + 1
                        print("[{}] {} : {} x {}, {}".format(index,feature_Desc,X_train.shape[0],X_train.shape[1],getTimeNow()))
                        print("    >> n_estimator:{}, learning_rate:{}, depth:{}, L2_leaf_reg:{}".format(p_n_estimators,p_learning_rate,p_depth,p_l2_leaf_reg))
                        params = {
                           
                        }                                
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
    getFinalResultForThisRound(prjRootFolder, bM, allDataSet)
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
    model = model.fit(X_train, y_train)  

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

    outputFolder=prjRootFolder+"Output//@best//"
    X_importance=bM.X_test
    X_all = allDataSet.iloc[:,5:bM.endat].values
    pk_all = allDataSet.iloc[:,0:3].values
    print(type(X_all))
    print(bM.endat)
    dtest = xgb.DMatrix(X_all, feature_names=bM.fname)
    ans = bM.model.predict(X_all)
    test = np.append(pk_all,np.array([ans]).transpose(),axis=1)
    resultColumnsName = np.append(allDataSet.columns.values[0:3],'Probability')
    result = pd.DataFrame(test,columns=resultColumnsName)
    #https://liuyanguu.github.io/post/2019/07/18/visualization-of-shap-for-xgboost/
    #https://shap.readthedocs.io/en/latest/
    #http://sofasofa.io/tutorials/shap_xgboost/
    #https://slundberg.github.io/shap/notebooks/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html
    #https://medium.com/ai-academy-taiwan/explain-your-machine-learning-model-by-shap-part-1-228fb2a57119
    explainer = shap.TreeExplainer(bM.model)
    shap_values = explainer.shap_values(X_importance)
    
    #嫌圖太大改這邊
    shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=5)
    shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=10)
    shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=20)
    shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=54)
    shap.summary_plot(shap_values, X_importance, plot_type='bar', feature_names=bM.fname, max_display=5)
    shap.summary_plot(shap_values, X_importance, plot_type='bar', feature_names=bM.fname, max_display=10)
    shap.summary_plot(shap_values, X_importance, plot_type='bar', feature_names=bM.fname, max_display=20)
    shap.summary_plot(shap_values, X_importance, plot_type='bar', feature_names=bM.fname, max_display=54)
    
    shap.force_plot(explainer.expected_value, shap_values[0,:],  bM.fname)
    
    #fim=pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('feature', ascending=False)    
    fs=pd.Series(bM.model.get_best_score()).sort_values(ascending=False)
    
    is_cat = (X_all[0,:] != float )

    cat_features_index = np.where(X_all)[0]
    print(2111)
    pool = Pool(X_all, ans, cat_features=cat_features_index, feature_names=bM.fname.tolist())
    pp = bM.model.plot_tree(
            tree_idx=0
    )

    pp.render('test-output/NelOrz', view=True)
    
    
    if save == 1:
       svpd = pd.DataFrame(shap_values)
       result.to_csv("{}//{}_Result.csv".format(outputFolder,bM.fileName))
       svpd.to_csv("{}//{}_ResultSHAP.csv".format(outputFolder,bM.fileName))
       fs.to_csv("{}//{}_FeatureImportance.txt".format(outputFolder,bM.fileName))
       graph = xgb.to_graphviz(bM.model, num_trees=1, **{'size': str(10)})
       graph.render(directory=outputFolder,filename=str(bM.fileName)+'_xgb.dot')
       #bM.model.save_model('{}_xgb.model'.format(bM.fileName))


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
    