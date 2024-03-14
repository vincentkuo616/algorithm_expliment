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
    
    #資料結構請參考上面 Test_Y190919_2
    prjRootFolder="C://Users//vincentkuo//Documents//vincent//"
    TestSet = pd.read_csv(prjRootFolder+"Y191028//TestData_A_Subset2.csv",encoding='utf-8')    
        
    '''
    https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
    '''    
    
    #loopStart
    
    logRoof()
    print("   [ test Loop ]") 
    logFloor()

    bestModelPath=""
    bestF1s=0
    bestX_test={}
    besty_test={}

    for featureSet in range (1,2):
        if featureSet==1 :
            test_subject="AllKNN" #僅供LOG檔名使用
            test_dev="AllKNN" #測試機器，用以比較不同機台同樣的設定跑速是否有差
            X_test = TestSet.iloc[:,6:71].values  
            y_test = TestSet.iloc[:,-1].values
            #y_test = TestSet.iloc[:,3].values
            feature_Desc="AllKNN"

        elif featureSet==2 :
            test_subject="AllKNNSMOTE" #僅供LOG檔名使用
            test_dev="AllKNNSMOTE" #測試機器，用以比較不同機台同樣的設定跑速是否有差
            TrainSet = pd.read_csv(prjRootFolder+"vincent//AllKNNSMOTE_FINAL.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,1:55].values
            y_train = TrainSet.iloc[:,0].values
            X_test = TestSet.iloc[:,5:59].values  
            y_test = TestSet.iloc[:,2].values
            feature_Desc="AllKNNSMOTE"
        
        else :
            test_subject="ADASYNENN" #僅供LOG檔名使用
            test_dev="ADASYNENN" #測試機器，用以比較不同機台同樣的設定跑速是否有差
            TrainSet = pd.read_csv(prjRootFolder+"vincent//ADASYNENN_FINAL.csv",encoding='utf-8')
            X_train = TrainSet.iloc[:,1:55].values
            y_train = TrainSet.iloc[:,0].values
            X_test = TestSet.iloc[:,5:59].values  
            y_test = TestSet.iloc[:,2].values
            feature_Desc="ADASYNENN"
            
        lstime=time.time()
        
        #TEST
        filePath = prjRootFolder+"Output//20191029112838.pkl"
        ar1 = xgBoost_testForLoop(X_test,y_test,filePath)

        
        letime=time.time()
        costTimeTrain=round(letime-lstime,2)
        print("  >> cost: "+str(costTimeTrain)+"s")
        
        F1Now=float(ar1[10]);
        if F1Now>bestF1s:
            bestF1s=F1Now
            bestModelPath=filePath
            bestX_test=X_test
            besty_test=y_test

    logRoof()
    xgBoost_testFortheBest(bestX_test,besty_test,bestModelPath)
    logFloor()
    # feature importance
    #plot_importance(bestModel)
    #plt.show()
    
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