# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:31:34 2020

@author: syn009
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, mstats
import pymannkendall as mk

pk='ITEM_NO'

def Cox_Stuart(list_c):

    lst=list_c.copy()
    n0=len(lst)
    if n0%2==1:
        del lst[int((n0-1)/2)]
    c=int(len(lst)/2)
    n_pos=n_neg=0
    for i in range(c):
        diff=lst[i+c]-lst[i]
        if diff>0:
            n_pos+=1
        elif diff<0:
            n_neg+=1
        else:
            continue
    n1=n_pos+n_neg
    if(n_pos<n_neg):
        p_value=stats.binom.cdf(n_pos,n1,0.5)
    else:
        p_value=stats.binom.cdf(n_neg,n1,0.5)
    '''
    n1=n_pos+n_neg
    k=min(n_pos,n_neg)
    p_value=2*stats.binom.cdf(k,n1,0.5)
    '''
    print('fall:%i, rise:%i, p-value:%f'%(n_neg, n_pos, p_value))

    returnString=''

    if n_pos>n_neg and p_value<0.05:
        returnString='increasing'
    elif n_neg>n_pos and p_value<0.05:
        returnString='decreasing'
    else:
        returnString='no trend'
    
    return [n_neg,n_pos,p_value,returnString]

from datetime import datetime
def getDatetimeStr():
    d = datetime.now()
    str='{:%Y%m%d%H%M%S}'.format(d)
    return str

prjRootFolder = r"D:\Temp\Vincent\臨時檔案"

readfile="test4CUSUM_Y220314"
data= pd.read_excel(f"{prjRootFolder}\\{readfile}.xlsx",encoding='utf-8')

dataWithGroup=data.groupby([pk])
lists=[]

x=2

if x==1:
    for index in dataWithGroup.groups.keys():
        theOne = dataWithGroup.get_group(index)
        if len(theOne)==1:
            continue 
        theOneSalesAmtList=theOne['DIFF'].values.tolist()
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(theOneSalesAmtList)
        #result=Cox_Stuart(theOneSalesAmtList)
        Mcode=theOne[pk].iloc[0]
        #result.append(Mcode)
        #print(result)
        #lists.append(result)
        #result=[]
        resultList=[trend,h,z,Tau,s,var_s,slope,intercept,Mcode]
        lists.append(resultList)
        resultList=[]
        
    OutputDF=pd.DataFrame(lists,columns=['MK_trend','MK_h','MK_z','MK_Tau','MK_s','MK_var_s','MK_slope','MK_intercept',pk])
    OutputDF.to_csv(f"{prjRootFolder}\\{readfile}_MK.csv",sep=",",index=False)


elif x==2:
    for index in dataWithGroup.groups.keys():
        theOne = dataWithGroup.get_group(index)
        if len(theOne)==1:
            continue
        theOneSalesAmtList=theOne['DIFF'].values.tolist()
        print(theOneSalesAmtList)
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(theOneSalesAmtList)
        result=Cox_Stuart(theOneSalesAmtList)
        Mcode=theOne[pk].iloc[0]
        result.append(Mcode)
        print(result)
        lists.append(result)
        result=[]
    OutputDF=pd.DataFrame(lists,columns=['COX_NEG','COX_POS','COX_P_value','COX_result',pk])
    OutputDF.to_csv(f"{prjRootFolder}\\{readfile}_Cox2.csv",sep=",",index=False)
    