from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import time 
from sklearn.externals import joblib
from sklearn import metrics   ###?算roc和auc
from sklearn.metrics import confusion_matrix
from datetime import datetime, timedelta
from collections import Counter




save = 1 #存檔開關

def main():
    test_subject="ADABOOST" #僅供LOG檔名使用
    test_dev="ADABOOST" #測試機器，用以比較不同機台同樣的設定跑速是否有差

    #資料結構請參考上面
    prjRootFolder="C://Users//vincentkuo//Documents//"
    TrainSet = pd.read_csv(prjRootFolder+"AllKNNSMOTE_8%.csv",encoding='utf-8')
    TestSet = pd.read_csv(prjRootFolder+"Test_Y190919_2.csv",encoding='utf-8')
    reportPath=prjRootFolder+"vincent//Report//adaboost_Vi_Y190927_"+test_subject+"_log"+getDatetimeStr()+".txt"
    
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
    
     #loop setting
    #========================================================
    #迭代次數 建議: [3-10]
    n_estimatorsMin=800 # Default=50
    n_estimatorsMax=1400
    n_estimatorsStep=50
    #分裂後損失函數大於此閾值才會長子結點，增加:避免overfitting 建議: [0-0.2]
    learning_rateMin=0.001
    learning_rateMax=0.31
    learning_rateStep=0.05
    #對於每棵樹隨機採樣的比例，降低:避免overfitting；過低:underfitting 建議: [0.5-0.9]
    random_stateMin=0
    random_stateMax=1
    random_stateStep=2
    #========================================================
    
    #loopStart
    
    logRoof()
    print("   [ test Loop ]") 
    logFloor()
    
    bM=bestModel("", "", 0, 0, {}, {}, xgb.core.Booster(), {}, 0)

    for featureSet in range (1,2):
        if featureSet==1 :
            endat=135
            feature_Desc="IV>=0.2"
            X_train = TrainSet.iloc[:,1:endat-2].values
            y_train = TrainSet.iloc[:,0].values
            X_test = TestSet.iloc[:,3:endat].values
            y_test = TestSet.iloc[:,2].values
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
        
        for p_n_estimators in np.arange (n_estimatorsMin, n_estimatorsMax, n_estimatorsStep):
            for p_learning_rate in np.arange (learning_rateMin, learning_rateMax, learning_rateStep):
                for p_random_state in np.arange (random_stateMin, random_stateMax, random_stateStep):
                    index = index + 1
                    print("[{}] {} : {} x {}, {}".format(index,feature_Desc,X_train.shape[0],X_train.shape[1],getTimeNow()))
                    print("    >> n_estimator:{}, learning_rate:{}, random_state:{}".format(p_n_estimators,p_learning_rate,p_random_state))
                    params = {
                       
                    }                                
                    plst = params.items()
                    timestp = getDatetimeStr()
                    filePath = prjRootFolder+"vincent//Output//"+timestp+".pkl"
                    lstime = time.time()
                    
                    #TRAIN
                    model = adaBoost_train(X_train,y_train,p_n_estimators, p_random_state, p_learning_rate,filePath)
                    #TEST
                    mS = adaBoost_testForLoop(X_test,y_test,filePath,fname)
                    
                    letime=time.time()
                    costTimeTrain=round(letime-lstime,2)
                    print("  >> cost: "+str(costTimeTrain)+"s")
                    resultStr="{},{},{},{},{},{},{},{},{},{}".format(test_dev,index,costTimeTrain,featureSet,p_n_estimators,p_random_state,p_learning_rate,mS.toString(),timestp,getTimeNow())
                    if save == 1:
                        with open(reportPath, "a") as myfile:
                            myfile.write(resultStr+"\n")
                        
                    if mS.f1s>bM.f1s:
                        bM=bestModel(filePath, timestp, index, mS.f1s, X_test, y_test, model, fname, endat)
                                    
    #playsound(prjRootFolder+'//Code//CNoc.mp3')
    logRoof()
    print(">> Loop End @ "+str(index))
    print(">> Best Sel @ {}".format(bM.index))
    adaBoost_testFortheBest(bM.X_test, bM.y_test, bM.filePath, bM.fname)
    #getFinalResultForThisRound(prjRootFolder, bM, allDataSet)
    logFloor()
    
def adaBoost_train(X_train,y_train,p_n_estimators, p_random_state, p_learning_rate,filePath):#train_data,nu,kernel,gamma
    model = AdaBoostClassifier(
                    n_estimators=p_n_estimators, 
                    random_state=p_random_state,
                    learning_rate=p_learning_rate
                    )
    model = model.fit(X_train, y_train)  
    with open(filePath, 'wb') as model_file:
        joblib.dump(model, model_file)
    return model

def adaBoost_testForLoop(X_test,y_test,filePath,fname):
    with open(filePath, 'rb') as f:
        model = joblib.load(f)
        predictY = model.predict(X_test)
        cm = confusion_matrix(y_test, predictY)
        mS = getCMresults(cm)
        print(colored('  The Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))
    return mS

def adaBoost_testFortheBest(X_test,y_test,filePath,fname):
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
    X_all = allDataSet.iloc[:,3:bM.endat].values
    pk_all = allDataSet.iloc[:,0:3].values
    dtest = xgb.DMatrix(X_all, feature_names=bM.fname)
    ans = bM.model.predict(dtest)
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
    shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=80)
    shap.summary_plot(shap_values, X_importance, plot_type='bar', feature_names=bM.fname, max_display=80)
    #shap.force_plot(explainer.expected_value, shap_values[0,:],  bM.fname)
    
    #fim=pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('feature', ascending=False)    
    fs=pd.Series(bM.model.get_fscore()).sort_values(ascending=False)
    xgb.plot_tree(bM.model)
    plt.show()

    if save == 1:
       svpd = pd.DataFrame(shap_values)
       result.to_csv("{}//{}_Result.csv".format(outputFolder,bM.fileName))
       svpd.to_csv("{}//{}_ResultSHAP.csv".format(outputFolder,bM.fileName))
       fs.to_csv("{}//{}_FeatureImportance.txt".format(outputFolder,bM.fileName))
       graph = xgb.to_graphviz(bM.model, num_trees=1, **{'size': str(10)})
       graph.render(directory=outputFolder,filename=str(bM.fileName)+'_xgb.dot')
       #bM.model.save_model('{}_xgb.model'.format(bM.fileName))
    
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
    