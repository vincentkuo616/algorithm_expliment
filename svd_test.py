# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:14:23 2021

@author: vincentkuo
"""


#coding=utf-8
from numpy import *
from numpy import linalg as la

'''載入測試資料集'''
def loadExData():
    return mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

'''以下是三種計算相似度的演算法，分別是歐式距離、皮爾遜相關係數和餘弦相似度,
注意三種計算方式的引數inA和inB都是列向量'''
def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))  #範數的計算方法linalg.norm()，這裡的1/(1+距離)表示將相似度的範圍放在0與1之間

def pearsSim(inA,inB):
    if len(inA)<3: return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]  #皮爾遜相關係數的計算方法corrcoef()，引數rowvar=0表示對列求相似度，這裡的0.5+0.5*corrcoef()是為了將範圍歸一化放到0和1之間

def cosSim(inA,inB):
    num=float(inA.T*inB)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom) #將相似度歸一到0與1之間

'''按照前k個奇異值的平方和佔總奇異值的平方和的百分比percentage來確定k的值,
後續計算SVD時需要將原始矩陣轉換到k維空間'''
def sigmaPct(sigma,percentage):
    sigma2=sigma**2 #對sigma求平方
    sumsgm2=sum(sigma2) #求所有奇異值sigma的平方和
    sumsgm3=0 #sumsgm3是前k個奇異值的平方和
    k=0
    for i in sigma:
        sumsgm3+=i**2
        k+=1
        if sumsgm3>=sumsgm2*percentage:
            return k

'''函式svdEst()的引數包含：資料矩陣、使用者編號、物品編號和奇異值佔比的閾值，
資料矩陣的行對應使用者，列對應物品，函式的作用是基於item的相似性對使用者未評過分的物品進行預測評分'''
def svdEst(dataMat,user,simMeas,item,percentage):
    n=shape(dataMat)[1]
    simTotal=0.0;ratSimTotal=0.0
    u,sigma,vt=la.svd(dataMat)
    k=sigmaPct(sigma,percentage) #確定了k的值
    sigmaK=mat(eye(k)*sigma[:k])  #構建對角矩陣
    xformedItems=dataMat.T*u[:,:k]*sigmaK.I  #根據k的值將原始資料轉換到k維空間(低維),xformedItems表示物品(item)在k維空間轉換後的值
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0 or j==item:continue
        similarity=simMeas(xformedItems[item,:].T,xformedItems[j,:].T) #計算物品item與物品j之間的相似度
        simTotal+=similarity #對所有相似度求和
        ratSimTotal+=similarity*userRating #用"物品item和物品j的相似度"乘以"使用者對物品j的評分"，並求和
    if simTotal==0:return 0
    else:return ratSimTotal/simTotal #得到對物品item的預測評分

'''函式recommend()產生預測評分最高的N個推薦結果，預設返回5個；
引數包括：資料矩陣、使用者編號、相似度衡量的方法、預測評分的方法、以及奇異值佔比的閾值；
資料矩陣的行對應使用者，列對應物品，函式的作用是基於item的相似性對使用者未評過分的物品進行預測評分；
相似度衡量的方法預設用餘弦相似度'''
def recommend(dataMat,user,N=5,simMeas=cosSim,estMethod=svdEst,percentage=0.9):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]  #建立一個使用者未評分item的列表
    if len(unratedItems)==0:return 'you rated everything' #如果都已經評過分，則退出
    itemScores=[]
    for item in unratedItems:  #對於每個未評分的item，都計算其預測評分
        estimatedScore=estMethod(dataMat,user,simMeas,item,percentage)
        itemScores.append((item,estimatedScore))
    itemScores=sorted(itemScores,key=lambda x:x[1],reverse=True)#按照item的得分進行從大到小排序
    return itemScores[:N]  #返回前N大評分值的item名，及其預測評分值