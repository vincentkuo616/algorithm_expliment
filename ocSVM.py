import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc  ## 算roc和auc
from sklearn.metrics import confusion_matrix
from sklearn import svm
import pandas as pd
#from scipy import stats

## 定義參數

p_nu = 0.07 # 0.02~0.2
#p_kernel = 'linear'
p_kernel = 'rbf'
#p_kernel = 'poly'
#p_kernel = 'sigmoid'
#p_kernel = 'precomputed'
p_gamma = 'scale'
if(p_gamma==0):
    p_gamma='scale'
#p_gamma = 0.0331502 # 0.005~0.02 0.016  0.034-40.91  0.0338-40.85  0.0335-40.94  0.0333-40.98  0.0332-41.02  0.0331-41.0
                  # 0.033152-41.09  0.03315-41.1  0.33148-41.09  0.0331502-41.108
p_gamma = 0.03315
#p_gamma = # 0.04-40.37  0.03-39.12  0.028-40.37  0.025-40.47  0.02505-40.49  0.03378-40.72  0.034-40.60  0.0335-40.83
        # 0.0334-40.76  0.0336-40.62  0.03355-40.67  0.03346-40.79  0.03352-40.79  0.03349-40.85  0.033492-40.86  0.0334922-40.85
        # :100-41.18  97-41.189  96-41.20  90-41.12  7:96-41.72  11:96-42.65  13:96-43.06


TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\FOr_OCSVM_TRAIN.csv",encoding='utf-8')
TestSet  = pd.read_csv(r"C:\Users\vincentkuo\Documents\FOr_OCSVM_TEST.csv",encoding='utf-8')
#TestSet  = pd.read_csv(r"C:\FOr_OCSVM_ALL.csv",encoding='utf-8')

n_samples = TrainSet.shape[0]  #Row Count
n_columns = TrainSet.shape[1]  #Column Count
#outliers_fraction=0.02 ## 要調!!!!!!!!!!!!!!!!!
#n_inliers = int( (1. - outliers_fraction) * n_samples ) ## 正常值筆數
#n_outliers = int( outliers_fraction * n_samples ) ## 異常值筆數

#Train_Data =  TrainSet.iloc[:,13:].values ## 所有特徵

def main():
    
    loaded_model, location = OneClassSVM_train(Train_Data,p_nu,p_kernel,p_gamma) ## 令模型為 loaded_model
    #location = 1
    #TestSet  = pd.read_csv(r"C:\FOr_OCSVM_ALL.csv",encoding='utf-8')
    #loaded_model = joblib.load(open("C://Users//syn007//Desktop//Vincent_Test//pkl//OCSVM190917004719.pkl", 'rb'))
    answer, ans_prob = OneClassSVM_test(Test_Data, loaded_model)
    checkFrequency(answer)
    yy, roc_auc = runROC(TestSet, ans_prob)
    plotconfusionMatrix(yy, answer)
    C_matrix = confusionMatrix(yy, answer)
    return C_matrix, roc_auc, location

def OneClassSVM_train(data1,p_nu,p_kernel,p_gamma):#train_data,nu,kernel,gamma

    trainSet = data1 ## 定義訓練資料為 trainSet
    clf = svm.OneClassSVM(nu=p_nu, gamma=p_gamma, kernel=p_kernel) ## 宣告模型
    clf.fit(trainSet) ## 訓練模型
    
    print('finish Train Model')
    print('************************************************************')
    ##########################################################################
    # 模型輸出落地
    #print('save model')
    time = str(datetime.datetime.now())
    hour = time[2:4]+time[5:7]+time[8:10]+time[11:13]+time[14:16]+time[17:19] #YYMMDDHHmmSS
    location = "C://Users//vincentkuo//Documents//OCSVM"+hour+".pkl" ## 檔案名稱
    #joblib.dump(clf, location, compress=0) ##
    print(location)
    return clf, location
    

#test

def OneClassSVM_test(Test_Data, loaded_model):
    
    print('Run testing data')
    # Paremeters for fitting the model
    # ====================== Parameter Description ================
    # n_estimators : int, optional (default=100)   #n_estimators:?????itree
    # max_samples : int or float, optional (default="auto")  #max_samples:???,???256
    # contamination : float in (0., 0.5), optional (default=0.1) #contamination:c(n)???0.1
    # max_features : int or float, optional (default=1.0)  #max_features:?????,???1
    # bootstrap:??Tree?,????????,?True???,?False????
    # n_jobs:fit?prdict???????
    # =============================================================
    
    scores_pred = loaded_model.decision_function(Test_Data) ## 回傳平均異常分數
    #print('*************')
    #threshold = stats.scoreatpercentile(scores_pred, 100*outliers_fraction ) ## 異常分數的閥值
    #print(scores_pred)
    #print(threshold)
    answer = loaded_model.predict(Test_Data) ## 回傳是否為異常值
    ans_prob =  loaded_model.score_samples(Test_Data) ## 回傳各樣本異常機率
    #print(ans_prob)
    return answer, ans_prob
    

##################################################################
#check frequency
def checkFrequency(answer):
    
    print('*************************************************************')
    print('check Frequency')
    arr = np.array(answer)
    print('Original Numpy Array : ' , arr)
    # Get a tuple of unique values & their frequency in numpy array
    uniqueValues, occurCount = np.unique(arr, return_counts=True)
    print("Unique Values : " , uniqueValues)
    print("Occurrence Count : ", occurCount)

##################################################################
# ROC
def runROC(TestSet, ans_prob):
    
    #print('*************************************************************')
    #print('Run ROC curve')
    # Compute ROC curve and ROC area for each class
    #把 Test的資料 從YN轉成 1  -1
    RealAns =  TestSet.iloc[:,2].values  ## Y值 UrgeMark
    RealAns[RealAns==1]='-1'
    RealAns[RealAns==0]='1'
    #RealAns = [str(i) for i in RealAns]
    fpr,tpr,threshold = roc_curve(RealAns, ans_prob, pos_label=1) ## 算真正率和假正率 (YY:正確值;score:機率值;pos_label;正常值)
    roc_auc = auc(fpr,tpr) ## 算auc的值
    #plt.figure(figsize=(8,8)) ## 定義lo的大小
    #plt.plot(fpr, tpr, color='darkorange',
     #        label='ROC curve (area = %0.3f)' % roc_auc) ## 作圖，並標示 AUC之值
    #plt.plot([0, 1], [0, 1], color='navy', linestyle='--') ## 做斜直線
    #plt.xlim([0.0, 1.0]) ## 定義 X 軸範圍
    #plt.ylim([0.0, 1.05]) ## 定義 Y 軸範圍
    #plt.xlabel('False Positive Rate') ## 定義 X 軸名稱
    #plt.ylabel('True Positive Rate') ## 定義 Y 軸名稱
    #plt.title('Receiver operating characteristic example') ## 定義 圖的名稱
    #plt.legend(loc="lower right") ## 定義 圖例位置
    #plt.grid(True) ## 框線顯示
    #plt.show()
    yy = [int(i) for i in RealAns]
    return yy, roc_auc

##################################################################
# Confusion Matrix
def plotconfusionMatrix(yy, answer):
    
    print('Run Confistion Matrix')
    
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plot_confusion_matrix(yy, answer,
     #                     title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    #plot_confusion_matrix(yy, answer, normalize=True,
     #                     title='Normalized confusion matrix')

    #plt.show()

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=np.array(['Y','N']), yticklabels=np.array(['Y','N']),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=41000, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
##################################################################
# Compute ACC
def confusionMatrix(yy, answer):
    
    C_matrix = confusion_matrix(yy,answer) ## 混淆矩陣 (yy:正確值;test2:預測值)
    print(C_matrix) ## show 混淆矩陣
    acc = round(100*(C_matrix[0,0]+C_matrix[1,1])/(C_matrix[0,0]+C_matrix[1,1]+C_matrix[0,1]+C_matrix[1,0]),5)
    print('ACC:', acc, '%')
    ppv = round(100*C_matrix[0,0]/(C_matrix[0,0]+C_matrix[1,0]),5)
    print('Precision:', ppv, '%')
    recall = round(100*C_matrix[0,0]/(C_matrix[0,0]+C_matrix[0,1]),5)
    print('Recall:', recall, '%')
    fScore = round(2/((1/ppv)+(1/recall)),5)
    print('F-Score:', fScore, '%')
    return C_matrix
##################################################################

#for i in range(1,21):
 #   p_nu = i*0.01
  #  for j in range(1,16):
   #     p_gamma = j/100+0.005
    #    tStart = time.time()
     #   C_matrix, roc_auc, location = main()
      #  TP = C_matrix[0,0]
       # TN = C_matrix[0,1]
        #FP = C_matrix[1,0]
        #FN = C_matrix[1,1]
        #tEnd = time.time()
        #str1= str(tEnd-tStart)+','+str(p_nu)+','+str(p_gamma)+','+str(roc_auc.round(3))+','+str((TP/(TP+TN)).round(4))+','+ str((TN/(TP+TN)).round(4))+','+str((FP/(FP+FN)).round(4))+','+str((FN/(FP+FN)).round(4))+','+str(TP)+','+str(TN)+','+str(FP)+','+str(FN)+','+str(location)
        #list2.append(str1)   ## 使用 append() 添加元素
        #with open("C://Users//syn007//Desktop//Vincent_Test//pkl//result.txt", "a") as myfile:
         #   myfile.write(str1+"\n")
        #print('================================  test End ====================================================')
for i in range(1,31):
    x=[i+2,i+3,i+4]
    TrainSet = pd.read_csv(r"C:\Users\vincentkuo\Documents\FOr_OCSVM_TRAIN.csv",encoding='utf-8')
    TestSet  = pd.read_csv(r"C:\Users\vincentkuo\Documents\FOr_OCSVM_TEST.csv",encoding='utf-8')
    TrainSet.drop(TrainSet.columns[x],axis=1,inplace=True)
    TestSet.drop(TestSet.columns[x],axis=1,inplace=True)
    Train_Data =  TrainSet.iloc[:,3:].values ## 所有特徵
    Test_Data =  TestSet.iloc[:,3:].values ## 所有特徵
    s = time.time()
    main()
    e = time.time()
    print('TIME:', e-s)