# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:54:52 2022

@author: vincentkuo
"""

from math import radians, cos, sin, asin, sqrt
from coord_system_trans import gcj02_to_wgs84, bd09_to_wgs84

def haversine(lon1, lat1, lon2, lat2): # 經度1 緯度1 經度2 緯度2
    # 將十進制度數轉化為弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine 公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    
    return c * r  #距離的單位為千米 (公里)

def haversine2(lon1, lat1, lon2, lat2, t1='wgs84', t2='wgs84'): # 經度1 緯度1 經度2 緯度2
    
    if(t1=='gcj02'):
        temp = gcj02_to_wgs84(lon1, lat1)
        lon1 = temp[0]
        lat1 = temp[1]
    if(t1=='bd09'):
        temp = bd09_to_wgs84(lon1, lat1)
        lon1 = temp[0]
        lat1 = temp[1]
    if(t2=='gcj02'):
        temp = gcj02_to_wgs84(lon2, lat2)
        lon2 = temp[0]
        lat2 = temp[1]
    if(t2=='bd09'):
        temp = bd09_to_wgs84(lon2, lat2)
        lon2 = temp[0]
        lat2 = temp[1]
    # 將十進制度數轉化為弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine 公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    
    return c * r  #距離的單位為千米 (公里)

def condition(data, n=1):
    #print(data)
    
    base_lon = data[data['基準'] == n].values[0][-2]
    base_lat = data[data['基準'] == n].values[0][-1]
    
    dataArray = data[data['基準'] == 0].values
    
    outputList = []
    for n in dataArray:
        lon = n[-2]
        lat = n[-1]
        distance = haversine(base_lon, base_lat, lon, lat)
        if base_lat>lat:
            distance = -distance
        if base_lon>lon:
            distance = -distance
            
        outputList.append([distance])
    return outputList




#import pandas as pd
#import numpy as np
#
##data = pd.read_excel('C://Users//vincentkuo//Downloads//北京經緯度A.xlsx',encoding='UTF-16')
#data = pd.read_excel('C://Users//vincentkuo//Downloads//北京經緯度A.xlsx')
#
#dataArray = data[data['基準'] == 0].values
#for n in range(1):
#    outputList = condition(data, n+1)
#    dataArray = np.append(dataArray, outputList, axis=1)
##print(dataArray)
#
#outputData = pd.DataFrame(dataArray)
#outputData.to_csv('C://Users//vincentkuo//Downloads//北京經緯度經緯度距離計算Result.csv',encoding='UTF-16',index=False)