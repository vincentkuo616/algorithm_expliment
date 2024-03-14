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
warnings.filterwarnings('ignore')
shap.initjs()

# prjRootFolder
#  ├ Code : Python Code
#  ├ Data : Dataset
#  ├ Output : Model
#  ├───@best : 本輪最佳解存在這
#  └ Report : log

save=0
#存檔開關，0=不存，1=存

def main():
   
    #資料結構請參考上面
    prjRootFolder="C://Users//vincentkuo//Documents//vincent_2//"
    #TrainSet = pd.read_csv(prjRootFolder+"Train_Data_rename.csv",encoding='utf-8')
    #TestSet = pd.read_csv(prjRootFolder+"Test_Data_rename.csv",encoding='utf-8')
    
    allDataSet = pd.read_csv(prjRootFolder+"ALL_Data_WB&VI&DA_Y191223.csv",encoding='utf-8')
    TrainSet = allDataSet.groupby("SAMPLE").get_group("TRAIN")
    TestSet = allDataSet.groupby("SAMPLE").get_group("TEST")
    #allDataSet = TrainSet.append(TestSet)
    
    #將資料分組
    TrainSet_A0 = pd.read_csv(prjRootFolder+"A0_Train_Data_Resampling_8%.csv",encoding='utf-8')
    TrainSet_A1 = TrainSet.groupby("GROUPA").get_group("A1")
    #TrainSet_B0 = pd.read_csv(prjRootFolder+"B0_Train_Data_Resampling_SE_13%.csv",encoding='utf-8')
    #TrainSet_B1 = pd.read_csv(prjRootFolder+"B0_Train_Data_Resampling_SE_8%",encoding='utf-8')
    TestSet_A0 = TestSet.groupby("GROUPA").get_group("A0")
    TestSet_A1 = TestSet.groupby("GROUPA").get_group("A1")

    A0=[7,8,67,78,88,102,106,107,109,112,126,135,171,213,216,217,368,375,380,385,391,403,407,428,440,475,479,494,497,499,508,512,514,517,529,531,534,536,539,551,552,553,558,559,565,566,567,569,578,579,636,637,638,639,644,645,646,647,653,654,655,660,662,663,669,671,674,676,686,704,720,721,722,727,744,745,756,757,758,759,763,764,768,777,798,803,821,828,829,861,876,883,913,914,915,927,935,672,481,574,703,572,573,533,532,560,496,495,493,706,530,934,170,169,863,9,862,166,918,218,877,917]
    A0=[7,8,67,78,88,102,106,107,109,112,126,135,171,213,216,368,375,380,385,391,403,407,428,440,475,479,494,497,499,508,517,529,531,534,536,539,551,552,553,558,559,565,566,567,569,579,636,637,638,639,644,645,646,647,653,654,655,660,662,663,669,671,674,676,686,704,720,721,722,727,744,745,756,757,758,759,763,764,768,777,798,803,821,828,829,861,876,883,913,914,915,927,935,672,481,574,703,572,573,533,532,560,496,495,493,706,530,934,170,169,863,9,862,166,918,218,877,917]
    A1=[8,68,78,79,86,88,99,102,107,108,112,130,140,252,273,368,374,379,383,384,386,391,395,404,409,428,439,463,478,479,495,499,509,511,512,513,514,529,530,531,534,536,539,551,552,563,565,566,567,569,578,579,636,637,638,639,644,645,646,647,653,654,655,660,662,669,671,674,676,704,717,721,722,723,744,745,756,757,758,769,796,800,805,807,821,824,831,844,848,858,859,863,881,883,553,558,559,672,574,703,572,673,573,497,533,532,560,496,493,706,494,7,861,935,934,169,9,862,876,915,166,918,877,921,916,922,897,159,866,896,907,869,888,153]
    sel_f = A1 # 挑選之特徵集
    '''
    A0.remove(217)
    A0.remove(512)
    A0.remove(514)
    A0.remove(578)
    '''
    A1.remove(68)
    feature_Y=2 #Y是第幾個欄位
    '''
    https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
    '''

    #loopStart
    
    logRoof()
    print("   [ test Loop ]") 
    logFloor()
    
    bM=bestModel("", "", 0, {}, {}, xgb.core.Booster(), {}, 0)

    for featureSet in range (1,2):
        if featureSet==1 :
            endat = sel_f
            feature_Desc="IV>=0.2"
            X_test = TestSet_A1.iloc[:,sel_f].values
            y_test = TestSet_A1.iloc[:,feature_Y].values
            y_test = transformYNto10(y_test)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=133)
        X_test2 = pca.fit_transform(X_test)
        print(X_test)
        print(X_test2)
        print(X_test.shape)
        print(X_test2.shape)

        fname=TrainSet.iloc[0:0,sel_f].columns.values
        print(fname)

        timestp = getDatetimeStr()
        filePath = "C://Users//vincentkuo//Documents//vincent_2//A1_80.51%.pkl" #A1_80.51% 20191231092032 TEST20191001153502 TEST20191002092521
        lstime = time.time()
        
        #TEST
        mS,model = xgBoost_testForLoop(X_test,y_test,filePath,fname)
        
        letime=time.time()
        costTimeTrain=round(letime-lstime,2)
        print("  >> cost: "+str(costTimeTrain)+"s")
            
        bM=bestModel(filePath, timestp, mS.f1s, X_test, y_test, model, fname, endat)
        
    #playsound(prjRootFolder+'//Code//CNoc.mp3')
    logRoof()
    xgBoost_testFortheBest(bM.X_test, bM.y_test, bM.filePath, bM.fname,bM.fileName)
    #getFinalResultForThisRound(prjRootFolder, bM, TestSet)
    logFloor()

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
    return mS,model

def xgBoost_testFortheBest(X_test,y_test,filePath,fname,fileName):
    with open(filePath, 'rb') as f:
        model = joblib.load(f)
        # pred the test data
        dtest = xgb.DMatrix(X_test, feature_names=fname)
        #dtest = xgb.DMatrix(X_test)
        ans = model.predict(dtest)
        test = np.array([ans]).transpose()
        result = pd.DataFrame(test)
        outputFolder="C://Users//vincentkuo//Documents//vincent//@best//"
        #result.to_csv("{}//{}_Result.csv".format(outputFolder,fileName))

        cm = confusion_matrix(y_test, (ans>0.5))
        mS=getCMresults(cm)
        print(colored('  The Best Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))        
        plot_importance(model, max_num_features=100)
        plt.show()
        
    return mS
    

def getFinalResultForThisRound(prjRootFolder, bM, TestSet):
    outputFolder=prjRootFolder+"@best//"
    X_importance=bM.X_test
    X_all = TestSet.iloc[:,bM.endat].values
    pk_all = TestSet.iloc[:,0:3].values
    dtest = xgb.DMatrix(X_all, feature_names=bM.fname)
    #dtest = xgb.DMatrix(X_all)
    ans = bM.model.predict(dtest)
    test = np.append(pk_all,np.array([ans]).transpose(),axis=1)
    resultColumnsName = np.append(TestSet.columns.values[0:3],'Probability')
    result = pd.DataFrame(test,columns=resultColumnsName)
    #https://liuyanguu.github.io/post/2019/07/18/visualization-of-shap-for-xgboost/
    #https://shap.readthedocs.io/en/latest/
    #http://sofasofa.io/tutorials/shap_xgboost/
    #https://slundberg.github.io/shap/notebooks/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html
    #https://medium.com/ai-academy-taiwan/explain-your-machine-learning-model-by-shap-part-1-228fb2a57119
    explainer = shap.TreeExplainer(bM.model)
    shap_values = explainer.shap_values(X_importance)
    #shap_values = explainer.shap_values(X_all)
    print('base value :',explainer.expected_value)
    
    #Dependence plot ARAMTMax_Cred_ALL_Ratio
    #shap.dependence_plot('ARAMT_L1MxMax_Ratio',shap_values,features=X_all,feature_names=bM.fname,interaction_index='ARAMTperYear_ALL')

    #Local interpretability 個別樣本的圖
    '''
    print('第0筆 data 預測為 True 的 Probability: [',ans[0],']')
    shap.force_plot(explainer.expected_value,shap_values[0,:],feature_names=bM.fname,matplotlib=True,figsize=(10,3),text_rotation=45)
    print('第1筆 data 預測為 True 的 Probability: [',ans[1],']')
    shap.force_plot(explainer.expected_value,shap_values[1,:],feature_names=bM.fname,matplotlib=True,figsize=(10,3),text_rotation=45)
    print('第2筆 data 預測為 True 的 Probability: [',ans[2],']')
    shap.force_plot(explainer.expected_value,shap_values[2,:],feature_names=bM.fname,matplotlib=True,figsize=(10,3),text_rotation=45)
    #fig = shap.force_plot(explainer.expected_value,shap_values[2,:],feature_names=bM.fname,matplotlib=True,figsize=(10,3),text_rotation=45,show=False)
    print('第12筆 data 預測為 True 的 Probability: [',ans[12],']')
    shap.force_plot(explainer.expected_value,shap_values[12,:],feature_names=bM.fname,matplotlib=True,figsize=(10,3),text_rotation=45,show=False)
    plt.savefig("C://Users//vincentkuo//Documents//vincent//@best//temp2.png")
    '''
    
    #fig = shap.image_plot(shap_values,X_all,show=False, width=20, aspect=0.2, hspace=0.2)
    #plt.savefig("C://Users//vincentkuo//Documents//vincent//@best//temp.png")
    
    #Sigmoid Function 
    #plotSigmoid()
    #嫌圖太大改這邊
    shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=80, show=False)
    plt.savefig("C://Users//vincentkuo//Documents//vincent//@best//temp2.png")
    plt.cla()
    shap.summary_plot(shap_values, X_importance, plot_type='bar', feature_names=bM.fname, max_display=80, show=False)
    plt.savefig("C://Users//vincentkuo//Documents//vincent//@best//temp3.png")
    #shap.force_plot(explainer.expected_value, shap_values[0,:],  feature_names=bM.fname)
    
    #fim=pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('feature', ascending=False)    
    #fs=pd.Series(bM.model.get_fscore()).sort_values(ascending=False)
    #xgb.plot_tree(bM.model)
    #plt.show()

    if save == 1:
       svpd = pd.DataFrame(shap_values,columns=bM.fname)
       #result.to_csv("{}//{}_Result.csv".format(outputFolder,bM.fileName))
       svpd.to_csv("{}//{}_ResultSHAP.csv".format(outputFolder,bM.fileName))
       #fs.to_csv("{}//{}_FeatureImportance.txt".format(outputFolder,bM.fileName))
       #graph = xgb.to_graphviz(bM.model, num_trees=1, **{'size': str(10)})
       #graph.render(directory=outputFolder,filename=str(bM.fileName)+'_xgb.dot')
       #bM.model.save_model('{}_xgb.model'.format(bM.fileName))
    
    #loaded_model = joblib.load('xgb.model')

def plotSigmoid():
    xs = np.linspace(-5,5,100)
    plt.xlabel('Shap value')
    plt.ylabel('Probability of UrgeMark="Y"')
    plt.title('Sigmoid Function')
    plt.plot(xs, 1/(1+np.exp(-xs)))
    
    new_ticks = np.linspace(-5,5,11)
    plt.xticks(new_ticks)
    plt.show()

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

def transformYNto10(yn):
    try:
        yn[yn=='Y']=1
        yn[yn=='N']=0
        yn = yn.astype(np.int64)
        return yn
    except:
        print("Something Wrong at transformYNto10")
    
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
    def __init__(self, filePath, fileName, f1s, Xt, yt, model, fname, endat):
        self.filePath=filePath
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