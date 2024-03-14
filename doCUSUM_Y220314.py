# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:30:29 2020

@author: vincentkuo
"""


import numpy as np
import pandas as pd
import time

#檔案位置
loc="D://文件//"
dataName="//test4CUSUM_Y220314.xlsx"
#存檔註記
saveFlag = 1
outputDataName="output4CUSUM_Y220314.csv"
#參數
threshold = 4
multiple = 0.5 #幾倍的STD

def main():
    
    data=readData(loc+dataName)
    output=pd.DataFrame()
    output=calculateLogic(data,output)
    
    if saveFlag==1:
        output.to_csv("D:\\文件\\"+outputDataName,index=False)
    else:
        print(output.head())

def calculateLogic(data,output=pd.DataFrame()):
    try:
        print('###################### Start Loop ########################')
        #By ItemNo Calculate
        dataGroup=data.groupby('ITEM_NO')
        #Loop 4 Calculate CUSUM
        for key in dataGroup.groups.keys():
            
            dataSelected = dataGroup.get_group(key)
            dataSelected = dataSelected.sort_values('Rank_Itemno')
            #Column 4 Calculate CUSUM
            x=dataSelected['GP_SPR_P'].values
            x_std=(x-x.mean())/x.std()
            
            #Change Point Start at Num
            start = -1
            #Change Points List
            changePoints=[]
            
            '''
            Loop 4 Define Change Points
            variable : x_std (from start+1 to end)
            K : Standard gap 4 Change Points
            threshold : Cumulation to the Threshold will return Change Point
            Return : changePoints List (Record 4 each Change Point)
            '''
            for l in np.arange(len(x_std)):
                changePoint = 'Default'
                if len(x_std[start+1:l])==0:
                    pass
                else:
                    #mean = x_std[start+1:].mean()
                    #std  = x_std[start+1:].std()
                    #mean = x_std[0:].mean()
                    #std  = x_std[0:].std()
                    mean = x_std[start+1:l].mean()
                    std  = x_std[start+1:l].std()
                    changePoint = cusum(x_std[start+1:l],mean=mean,K=std*multiple,threshold=threshold)
                if type(changePoint)==np.int32 or len(str(changePoint))==1:
                    changePoint=changePoint + start +1 #Real Index
                    changePoints.append(changePoint)
                    start=changePoint
            '''
            Loop 4 Use Change Points List to Create New Columns "SPLIT" & "CP"
            "SPLIT" means the line split by change points
            "CP" means the Change Points Flag
            
            '''
            lableList=[]
            lableCP=[]
            index=0
            for i in range(len(x)):
                changePoints.append(len(x))
                changePoint = changePoints[index]
                if i < changePoint:
                    lableList.append(index)
                    lableCP.append("")
                elif i==changePoint:
                    lableList.append(index)
                    lableCP.append('CP')
                else:
                    lableList.append(index+1)
                    lableCP.append("")
                    index+=1
            #output=output.append(pd.DataFrame({'ITEM_NO':dataSelected['ITEM_NO'].values, 'SLIP_NO':dataSelected['SLIP_NO'].values,'SLIP_LINE_NO':dataSelected['SLIP_LINE_NO'].values,'x':dataSelected['GP_SPR_P'].values, 'SPLIT':lableList, 'CP':lableCP, 'Rank_Itemno':dataSelected['Rank_Itemno'].values}))
            dataSelected['SPLIT']=lableList
            dataSelected['CP']=lableCP
            output=output.append(dataSelected)
    except:
        print('Something Wrong at CalculateLogic')
    return output

def cusum(x,mean=0,K=0,threshold=0):
    """Tabular CUSUM per Montgomery,D. 1996 "Introduction to Statistical Process Control" p318 
    x    : series to analyze
    mean : expected process mean
    K    : reference value, allowance, slack value-- suggest K=1/2 of the shift to be detected.

    Returns:
    x  Original series
    Cp positive CUSUM
    Cm negative CUSUM
    """
    Cp=(x*0).copy()
    Cm=Cp.copy()
    for ii in np.arange(len(x)):
        if ii == 0:
            Cp[ii]=0
            Cm[ii]=0
        else:
            Cp[ii]=np.max([0,x[ii]-(mean+K)+Cp[ii-1]])
            Cm[ii]=np.max([0,(mean-K)-x[ii]+Cm[ii-1]])
        if Cp[ii]>threshold or Cm[ii]>threshold:
            return ii
    return({'x':x, 'Cp': Cp, 'Cm': Cm})

def readData(root):
    try:
        if root[-3:]=='csv':
            data = pd.read_csv(root,encoding='utf-16')
        elif root[-3:]=='lsx':
            data = pd.read_excel(root,encoding='utf-8')
        else:
            data = pd.read_excel(root,encoding='utf-8')
    except:
        print("Data parsing failed")
    return data

if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")