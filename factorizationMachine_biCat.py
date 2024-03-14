# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:34:15 2021

@author: vincentkuo
"""


import random
from numpy import *
from random import normalvariate  # 正態分佈
from datetime import datetime
import pandas as pd
import numpy as np

# 資料集切分
def loadData(fileName,ratio):    # ratio，訓練集與測試集比列
    trainingData = []
    testData = []
    with open(fileName) as txtData:
        lines = txtData.readlines()
        for line in lines:
            lineData = line.strip().split(',')
            if random.random() < ratio:           #資料集分割比例
                trainingData.append(lineData)     #訓練資料集列表
            else:
                testData.append(lineData)         #測試資料集列表
            np.savetxt('C:\\Users\\vincentkuo\\Documents\\diabetes_train.txt', trainingData, delimiter=',',fmt='%s')
            np.savetxt('C:\\Users\\vincentkuo\\Documents\\diabetes_test.txt', testData, delimiter=',',fmt='%s')
    return trainingData,testData

diabetes_file = 'C:\\Users\\vincentkuo\\Documents\\diabetes.csv'
trainingData, testData = loadData(diabetes_file,0.8)

# FM二分類
# diabetes皮馬人糖尿病資料集
# coding:UTF-8


# 處理資料
def preprocessData(data):
    feature = np.array(data.iloc[:, :-1])  # 取特徵(8個特徵)
    label = data.iloc[:, -1].map(lambda x: 1 if x == 1 else -1)  # 取標籤並轉化為 +1，-1

    # 將陣列按行進行歸一化
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)  # 特徵的最大值，特徵的最小值
    feature = (feature - zmin) / (zmax - zmin)
    label = np.array(label)

    return feature, label


def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


# 訓練FM模型
def FM(dataMatrix, classLabels, k, iter, alpha):
    '''
    :param dataMatrix:  特徵矩陣
    :param classLabels: 標籤矩陣
    :param k:           v的維數
    :param iter:        反覆運算次數
    :return:            常數項w_0, 一階特徵係數w, 二階交叉特徵係數v
    '''
    # dataMatrix用的是matrix, classLabels是列表
    m, n = shape(dataMatrix)  # 矩陣的行列數，即樣本數m和特徵數n

    # 初始化參數
    w = zeros((n, 1))  # 一階特徵的係數
    w_0 = 0  # 常數項
    v = normalvariate(0, 0.2) * ones((n, k))  # 即生成輔助向量(n*k)，用來訓練二階交叉特徵的係數

    for it in range(iter):
        for x in range(m):  # 隨機優化，每次只使用一個樣本
            # 二階項的計算
            inter_1 = dataMatrix[x] * v  # 每個樣本(1*n)x(n*k),得到k維向量（FM化簡公式大括弧內的第一項）
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # 二階交叉項計算，得到k維向量（FM化簡公式大括弧內的第二項）
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.  # 二階交叉項計算完成（FM化簡公式的大括弧外累加）

            p = w_0 + dataMatrix[x] * w + interaction  # 計算預測的輸出，即FM的全部項之和
            tmp = 1 - sigmoid(classLabels[x] * p[0, 0])  # tmp反覆運算公式的中間變數，便於計算
            w_0 = w_0 + alpha * tmp * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] + alpha * tmp * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] + alpha * tmp * classLabels[x] * (
                                dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

        # 計算損失函數的值
        if it % 10 == 0:
            loss = getLoss(getPrediction(mat(dataMatrix), w_0, w, v), classLabels)
            print("第{}次反覆運算後的損失為{}".format(it, loss))

    return w_0, w, v


# 損失函數
def getLoss(predict, classLabels):
    m = len(predict)
    loss = 0.0
    for i in range(m):
        loss -= log(sigmoid(predict[i] * classLabels[i]))
    return loss


# 預測
def getPrediction(dataMatrix, w_0, w, v):
    m = np.shape(dataMatrix)[0]
    result = []
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # multiply對應元素相乘
        # 完成交叉項
        interaction = np.sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 計算預測的輸出
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result


# 評估預測的準確性
def getAccuracy(predict, classLabels):
    m = len(predict)
    allItem = 0
    error = 0
    for i in range(m):  # 計算每一個樣本的誤差
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem


if __name__ == '__main__':
    trainData = 'C:\\Users\\vincentkuo\\Documents\\diabetes_train.txt'
    testData = 'C:\\Users\\vincentkuo\\Documents\\diabetes_test.txt'
    train = pd.read_csv(trainData)
    test = pd.read_csv(testData)
    dataTrain, labelTrain = preprocessData(train)
    dataTest, labelTest = preprocessData(test)
    date_startTrain = datetime.now()

    print("開始訓練")
    w_0, w, v = FM(mat(dataTrain), labelTrain, 4, 100, 0.01)
    print("w_0:", w_0)
    print("w:", w)
    print("v:", v)
    predict_train_result = getPrediction(mat(dataTrain), w_0, w, v)  # 得到訓練的準確性
    print("訓練準確性為：%f" % (1 - getAccuracy(predict_train_result, labelTrain)))
    date_endTrain = datetime.now()
    print("訓練耗時為：%s" % (date_endTrain - date_startTrain))

    print("開始測試")
    predict_test_result = getPrediction(mat(dataTest), w_0, w, v)  # 得到訓練的準確性
    print("測試準確性為：%f" % (1 - getAccuracy(predict_test_result, labelTest)))
