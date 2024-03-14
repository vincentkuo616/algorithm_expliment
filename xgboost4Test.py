import pickle
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
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from termcolor import colored
import math
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


# prjRootFolder
#  ├ Code : Python Code
#  ├ Data : Dataset
#  ├ Output : Model
#  └ Report : log

def main():
    test_subject="XGBOOST" #僅供LOG檔名使用
    
    #資料結構請參考上面
    prjRootFolder="C://Users//vincentkuo//Documents//vincent//"
    TrainSet = pd.read_csv(prjRootFolder+"Y191028//TrainData_B_Subset3.csv",encoding='utf-8')
    TestSet = pd.read_csv(prjRootFolder+"Y191028//TestData_B_Subset3.csv",encoding='utf-8')    
    reportPath=prjRootFolder+"Report//Test_xgBoost_Vi_Y191029_"+test_subject+"_log"+getDatetimeStr()+".txt"
    

    titleStr="test_dev,index,costTime,featureSet,max_depth,gamma,subsample,scale_pos_weight,eta,min_child_weight,estimators,TN,FP,FN,TP,Precision,Recall,Specificity,Sensitivity,Accuracy,F1-Score,fileName,TimeStamp"
    with open(reportPath, "a") as myfile:
        myfile.write(titleStr+"\n")
        
    '''
    https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
    '''
    index=0
    
    #loop setting
    #========================================================
    #最大深度 建議: [3-10]
    depthMin=5 #這樣寫是固定3
    depthMax=12
    depthStep=2
    #分裂後損失函數大於此閾值才會長子結點，增加:避免overfitting 建議: [0-0.2]
    gammaMin=0.2
    gammaMax=0.3
    gammaStep=0.1
    #對於每棵樹隨機採樣的比例，降低:避免overfitting；過低:underfitting 建議: [0.5-0.9]
    subsampleMin=0.7
    subsampleMax=0.91
    subsampleStep=1
    #colsample_bytree 控制每顆樹隨機採樣的列數的佔比 建議: [0.5-0.9]
    cbMin=0.7
    cbMax=0.91
    cbStep=1
    #learning rate
    etaMin=0.2
    etaMax=0.4
    etaStep=0.1
    #min_child_weight 決定最小葉子節點樣本權重和，加權和低於這個值時，就不再分裂產生新的葉子節點 建議: [1]
    mcwMin=1
    mcwMax=3
    mcwStep=1
    #boost迭代次數
    itersMin=900
    itersMax=1200
    itersStep=50
    #========================================================

    
    
    #loopStart
    
    logRoof()
    print("   [ test Loop ]") 
    logFloor()

    bestModelPath=""
    bestIndex=0
    bestF1s=0
    bestX_test={}
    besty_test={}

    for featureSet in range (1,2):
        if featureSet==1 :
            endat=71
            feature_Desc="B_2Cluster"
            test_dev="B_2Cluster"
            TrainSet = pd.read_csv(prjRootFolder+"Y191028//TrainData_AllKNNSMOTE_Y191030_new.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,6:endat].values
            y_train = TrainSet.iloc[:,5].values
            X_test = TestSet.iloc[:,6:endat].values
            y_test = TestSet.iloc[:,5].values

        elif featureSet==2 :
            endat=61
            feature_Desc="SMOTE_ALLKNN"
            TrainSet = pd.read_csv(prjRootFolder+"vincent//Y191008//TrainData_SMOTE_ALLKNN_Y191008.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,3:endat].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,3:endat].values  
            y_test = TestSet.iloc[:,2].values
            
        elif featureSet==3 :
            endat=61
            feature_Desc="ALLKNN_SMOTE"
            test_dev="ALLKNN_SMOTE"
            TrainSet = pd.read_csv(prjRootFolder+"vincent//Y191008//TrainData_AllKNNSMOTE_Y191008.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,3:endat].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,3:endat].values  
            y_test = TestSet.iloc[:,2].values
            
        elif featureSet==4 :
            endat=61
            feature_Desc="ALLKNN"
            test_dev="ALLKNN"
            TrainSet = pd.read_csv(prjRootFolder+"vincent//Y191008//TrainData_AllKNN_Y191008.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,3:endat].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,3:endat].values  
            y_test = TestSet.iloc[:,2].values
            
        elif featureSet==5 :
            endat=72
            feature_Desc="SMOTE_ENN_NEW"
            test_dev="SMOTE_ENN_NEW"
            TrainSet = pd.read_csv(prjRootFolder+"vincent//Y191008//TrainData_SM_ENN_Y191008.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,14:endat].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,14:endat].values  
            y_test = TestSet.iloc[:,2].values
            
        elif featureSet==6 :
            endat=72
            feature_Desc="SMOTE_ALLKNN_NEW"
            test_dev="SMOTE_ALLKNN_NEW"
            TrainSet = pd.read_csv(prjRootFolder+"vincent//Y191008//TrainData_SMOTE_ALLKNN_Y191008.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,14:endat].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,14:endat].values  
            y_test = TestSet.iloc[:,2].values
            
        elif featureSet==7 :
            endat=72
            feature_Desc="ALLKNN_SMOTE_NEW"
            test_dev="ALLKNN_SMOTE_NEW"
            TrainSet = pd.read_csv(prjRootFolder+"vincent//Y191008//TrainData_AllKNNSMOTE_Y191008.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,14:endat].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,14:endat].values  
            y_test = TestSet.iloc[:,2].values
            
        elif featureSet==8 :
            endat=72
            feature_Desc="ALLKNN_NEW"
            test_dev="ALLKNN_NEW"
            TrainSet = pd.read_csv(prjRootFolder+"vincent//Y191008//TrainData_AllKNN_Y191008.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,14:endat].values
            y_train = TrainSet.iloc[:,2].values
            X_test = TestSet.iloc[:,14:endat].values  
            y_test = TestSet.iloc[:,2].values
        
        for p_depth in np.arange (depthMin, depthMax, depthStep): #max_depth
            for p_gamma in np.arange(gammaMin, gammaMax, gammaStep): #gamma
                for p_ss in np.arange(subsampleMin, subsampleMax, subsampleStep):
                    for p_cb in np.arange(cbMin, cbMax, cbStep):
                        for p_eta in np.arange (etaMin, etaMax, etaStep): 
                            for p_mcw in np.arange (mcwMin, mcwMax, mcwStep):
                                for iters in np.arange (itersMin, itersMax, itersStep):
                                    index = index + 1
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
                                        #'nthread': 4,
                                        'scale_pos_weight': 1, #如果出現嚴重的不平衡，則應使用大於0的值，因為它有助於加快收斂速度。 負樣本個數/正樣本個數 
                                        #上面這條有的人說可以調，有的人說調了沒用，所以請自行參考斟酌
                                    }
                                                    
                                    plst = params.items()
                                    #num_rounds = 1000
                                    dtrain = xgb.DMatrix(X_train, y_train)
                                    timestp=getDatetimeStr()
                                    filePath=prjRootFolder+"Output//"+timestp+".pkl"
                                    lstime=time.time()
                                    
                                    #TRAIN
                                    model = xgBoost_train(plst, dtrain, iters, filePath)
                                    #TEST
                                    ar1 = xgBoost_testForLoop(X_test,y_test,filePath)
                
                                    
                                    letime=time.time()
                                    costTimeTrain=round(letime-lstime,2)
                                    print("  >> cost: "+str(costTimeTrain)+"s")
                                    
                                    resultStr="{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(test_dev,index,costTimeTrain,featureSet,p_depth,p_gamma,p_ss,p_cb,p_eta,p_mcw,iters,ar1[0],ar1[1],ar1[2],ar1[3],ar1[4],ar1[5],ar1[6],ar1[7],ar1[8],ar1[9],timestp,getTimeNow())
                
                                    #resultlist.append(resultStr)  
                                    with open(reportPath, "a") as myfile:
                                        myfile.write(resultStr+"\n")
                                    
                                    F1Now=float(ar1[10]);
                                    if F1Now>bestF1s:
                                        bestF1s=F1Now
                                        bestIndex=index
                                        bestModelPath=filePath
                                        bestX_test=X_test
                                        besty_test=y_test
                                    
                                        
    logRoof()
    print(">> Loop End @ "+str(index))
    print(">> Best Sel @ {}".format(bestIndex))
    xgBoost_testFortheBest(bestX_test,besty_test,bestModelPath)
    logFloor()
    # feature importance
    #plot_importance(bestModel)
    #plt.show()
    

def xgBoost_train(plst, dtrain, num_rounds,filePath):#train_data,nu,kernel,gamma
    model = xgb.train(plst, dtrain, num_rounds)
    with open(filePath, 'wb') as model_file:
        pickle.dump(model, model_file)
    return model


def xgBoost_testForLoop(X_test,y_test,filePath):
    with open(filePath, 'rb') as f:
        model = pickle.load(f)
        # pred the test data
        dtest = xgb.DMatrix(X_test)
        ans = model.predict(dtest)
        cm = confusion_matrix(y_test, (ans>0.5))
        arr=getCMresults(cm)
        
        print(colored('  The Confusion Matrix is: ', 'red'),'\n', cm)
        ACC = round((cm[0,0] + cm[1,1])*100/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1]),2)
        print(colored('  >> ACC = ', 'blue'), colored(ACC, 'blue'), colored("%", 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(arr[10], 'blue'), colored("%", 'blue'))
        
    return arr

def xgBoost_testFortheBest(X_test,y_test,filePath):
    with open(filePath, 'rb') as f:
        model = pickle.load(f)
        # pred the test data
        dtest = xgb.DMatrix(X_test)
        ans = model.predict(dtest)
        cm = confusion_matrix(y_test, (ans>0.5))
        arr=getCMresults(cm)
        
        print(colored('  The Best Confusion Matrix is: ', 'red'),'\n', cm)
        ACC = round((cm[0,0] + cm[1,1])*100/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1]),2)
        print(colored('  >> Best ACC = ', 'blue'), colored(ACC, 'blue'), colored("%", 'blue'))
        print(colored('  >> Best F1-score = ', 'blue'), colored(arr[10], 'blue'), colored("%", 'blue'))
        plot_importance(model, max_num_features=100)
        plt.show()
    return arr[10]
    

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
    print(cm)
    tn=C_matrix[0, 0]
    fp=C_matrix[0, 1]
    fn=C_matrix[1, 0]
    tp=C_matrix[1, 1]
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision= tp / (tp + fp)
    accuracy=(tp + tn) / (tp + fn + fp + tn)
    recall= tp / (tp+fn)
    f1s=str(round(2*precision*recall / (precision + recall),6))
    f1sp=str(round(2*precision*recall*100 / (precision + recall),6))
    return np.array([tn,fp,fn,tp,precision,recall,specificity,sensitivity,accuracy,f1s,f1sp])
    
    

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
    
if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")    