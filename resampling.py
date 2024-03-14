import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.over_sampling import ADASYN ,SMOTE
from imblearn.under_sampling import AllKNN,EditedNearestNeighbours
import time



def main():
    #cost 
    prjRootFolder="C://Users//vincentkuo//Documents//vincent_TW//"
    allData_csv = "A0_ALL_DATA_orderbyAsetIV.csv"    
    isSave = 1 #1=要存檔
    x_Start = 11
    x_end = None
    y_col = 2
    TrainSet_Name = "A0_ALL_DATA_orderbyAsetIV_8%"
    #load Data
    x_list,y_list,column_Name= readData_splitTrainTest(prjRootFolder+allData_csv,x_Start,x_end,y_col,'GROUPA','A0','utf-8')

    #check  N Y distribution
    checkYFreq(y_list,'Raw Data')
    #Resampling
    x_smote,y_smote = smote(x_list,y_list,8,0.077,None)#x_list,y_list,k_neighbors,ratio,kind
    x_enn,y_enn     = enn(10,x_smote,y_smote)#k_neighbors,x_list,y_list
    
    #x_sme,y_sme = smoteEnn(x_list,y_list,87,0.08)
    #x_ada,y_ada     = adasyn(0.3,x_list,y_list)
    #x_knn,y_knn     = allKNN(8,x_list,y_list)
    #x_smote,y_smote = smote(x_knn,y_knn,8,0.09,None)#x_list,y_list,k_neighbors,ratio,kind
    #x_stk,y_stk     = smoteTomek(87,0.2,x_list,y_list)
    #mData = mergeData(x_enn,y_enn)
    #saveCSV(isSave,mData,prjRootFolder,'testResampling')
    
    checkYFreq(y_enn,'ReSampling')
    #save Data
    mergeAndSave(x_enn,y_enn,column_Name,isSave,prjRootFolder,TrainSet_Name,x_Start,x_end,y_col)

'''
    loading Data
    param : path=檔案路徑
            x_start=從第幾個欄位開始
            x_end=從第幾個欄位結束，全部用請打None，
            y=y是哪一格
            group = GROUPA or GROUPB
            groupCode = A0 A1 B0 B1
            encoding=編碼方式，預設utf-8
'''
def readData_splitTrainTest(path,x_start,x_end,y,group,groupCode,encoding):
    if(encoding is None):
        encoding='utf-8'
    data = pd.read_csv(path,encoding=encoding)  
    n_samples = data.shape[0]  #Row Count
    n_columns = data.shape[1]  #Column Count
    print("Row Cnt=",n_samples,"Columns=",n_columns)
    groupData = data.groupby(group).get_group(groupCode)
    groupData_Train= groupData.groupby("SAMPLE").get_group("TRAIN")
    print(group+"_"+groupCode+": "+"Row Cnt=",groupData.shape[0],"Columns=",groupData.shape[1])
    
    #回傳 x_list,y_list跟欄位名稱
    if(x_end is None):
        return groupData_Train.iloc[:,x_start:],groupData_Train.iloc[:,y],data.columns
    else:
        return groupData_Train.iloc[:,x_start:x_end],groupData_Train.iloc[:,y],data.columns
'''
    檢查Y占比
    param:
          y_list= y 
          title = 呈顯用
'''
def checkYFreq(y_list,title):
    if(title is None):
        title=''
    print('*****  check Frequency '+title +'  *********')
    arr = np.array(y_list)
    # Get a tuple of unique values & their frequency in numpy array
    uniqueValues, occurCount = np.unique(arr, return_counts=True)
    print("Unique Values : " , uniqueValues)
    print("Occurrence Count : ", occurCount)
    print("Y Rate = " ,occurCount[1]/(occurCount[0]+occurCount[1]))

def mergeData(x_list,y_list):
    mergeData = np.column_stack([y_list,x_list])
    return mergeData

def saveCSV(isSave,Data,path,fileName):
    if(isSave==1):
        temp = pd.DataFrame(Data)
        temp.to_csv(path+"//"+fileName+".csv")
'''儲存後會有欄位名稱
    Parameter:
        x_list
        y_list
        column_Name:欄位對應的中文 List
        isSave=存檔開關 , 1=存檔
        path=存檔路徑
        fileName=檔名
        x_start=從第幾個欄位開始
        x_end=從第幾個欄位結束，全部用請打None，
        y=y是哪一格
'''
def mergeAndSave(x_list,y_list,column_Name,isSave,path,fileName,x_start,x_end,y):
    print(x_list)
    print('====================')
    if(isSave==1):
        mergeData = np.column_stack([y_list,x_list])        
        for i  in range(0,x_start):
            if(i != y):
                mergeData=np.insert(mergeData,i,3,axis=1)
        temp = pd.DataFrame(mergeData,columns=column_Name)
        temp.to_csv(path+"//"+fileName+".csv",index=False)
'''
    Parameter:
        x_list
        y_list
        k_neighbor = 調整Y點附近有幾個鄰居(調整生成假數據的寬度)
        ratio = 調整Y占比
        kind = SMOTE方法，預設是空的，有 borderline1  borderline2  svm ,預設為 regular
        #(sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=1, m_neighbors=10, svm_estimator=None, out_step=0.5)
'''
def smote(x_list,y_list,k_neighbor,ratio,kind):
    if(kind is None):
        kind='regular'
    sm = SMOTE(n_jobs=-1,k_neighbors=k_neighbor,ratio=ratio,kind=kind)
    x_resampled, y_resampled = sm.fit_sample(x_list, y_list)
    checkYFreq(y_resampled,'smote')
    return x_resampled,y_resampled
'''
    Parameter:        
        k_neighbor = 調整X點附近有幾個鄰居
        x_list
        y_list
'''
def enn(n_neighbors,x_list,y_list):    
    enn = EditedNearestNeighbours(n_neighbors=n_neighbors)
    x_ENN_res, y_ENN_res=enn.fit_resample(x_list,y_list)
    checkYFreq(y_ENN_res,'enn')
    return x_ENN_res, y_ENN_res

'''
    Parameter:
        x_list
        y_list
        sampling_strategy = 調整Y占比 0.6版本後會移除此參數
        random_state = SEED(不要叫我翻譯種子)
'''
def smoteEnn(x_list,y_list,random_state,sampling_strategy):

    sme = SMOTEENN(random_state=random_state,sampling_strategy=sampling_strategy)
    X_res, y_res = sme.fit_resample(x_list, y_list)
    checkYFreq(y_res,'smoteEnn')
    return X_res,y_res
'''
    Parameter:
        ratio: Y 占比
        x_list
        y_list
'''
def adasyn(ratio,x_list,y_list):
    ada=ADASYN(ratio=ratio)
    x_ada_res,y_ada_res=ada.fit_resample(x_list,y_list)
    checkYFreq(y_ada_res,'adasyn')
    return x_ada_res,y_ada_res
'''
    Parameter:        
        k_neighbor 
        x_list
        y_list
'''
def allKNN(n_neighbors,x_list,y_list):
    oKNN=AllKNN(n_jobs=-1, n_neighbors=n_neighbors ,allow_minority =True)
    X_ada_res,y_ada_res=oKNN.fit_resample(x_list,y_list)
    checkYFreq(y_ada_res,'allKNN')
    return X_ada_res,y_ada_res
'''
    Parameter:        
        random_state =SEED
        ratio = Y占比
        x_list
        y_list
'''
def smoteTomek(random_state,ratio,x_list,y_list):
    sm = SMOTETomek(n_jobs=-1,random_state=random_state,ratio=ratio)
    x_resampled, y_resampled = sm.fit_sample(x_list, y_list)
    checkYFreq(y_resampled,'smoteTomek')
    return x_resampled, y_resampled

if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")  
    
    
    
'''
do not nun :  import this
████████▀▀░░░░░░░░░░░░░░░░░░░▀▀████████
██████▀░░░░░░░░░░░░░░░░░░░░░░░░░▀██████
█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█████
████░░░░░▄▄▄▄▄▄▄░░░░░░░░▄▄▄▄▄▄░░░░░████
████░░▄██████████░░░░░░██▀░░░▀██▄░░████
████░░███████████░░░░░░█▄░░▀░░▄██░░████
█████░░▀▀███████░░░██░░░██▄▄▄█▀▀░░█████
██████░░░░░░▄▄▀░░░████░░░▀▄▄░░░░░██████
█████░░░░░█▄░░░░░░▀▀▀▀░░░░░░░█▄░░░█████
█████░░░▀▀█░█▀▄▄▄▄▄▄▄▄▄▄▄▄▄▀██▀▀░░█████
██████░░░░░▀█▄░░█░░█░░░█░░█▄▀░░░░██▀▀▀▀
▀░░░▀██▄░░░░░░▀▀█▄▄█▄▄▄█▄▀▀░░░░▄█▀░░░▄▄
▄▄▄░░░▀▀██▄▄▄▄░░░░░░░░░░░░▄▄▄███░░░▄██▄
██████▄▄░░▀█████▀█████▀██████▀▀░░▄█████
██████████▄░░▀▀█▄░░░░░▄██▀▀▀░▄▄▄███▀▄██
███████████░██░▄██▄▄▄▄█▄░▄░████████░███
'''