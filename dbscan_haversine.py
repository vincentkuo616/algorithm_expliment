# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:03:48 2023

@author: vincentkuo
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

df = pd.read_excel("C:\\Users\\vincentkuo\\Downloads\\signLocationByApp\\APP簽到行為模式_經緯度資料.xlsx")

column = ['員編','出勤日','簽到退','attendance_lat(WGS)','attendance_lng(WGS)']

df = df[column]
print(df.head(5))

dataGroup=df.groupby('員編')

kms_per_radian = 6371.0088
#dbscan 參數:
epsilon_km = 0.5 #epsilon 單位是公里
min_samples = 2 #minpts

output = pd.DataFrame()

for name, group in dataGroup:
    dataSelected = group[['attendance_lat(WGS)', 'attendance_lng(WGS)']]
    
    epsilon = epsilon_km / kms_per_radian
    
    model = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    db = model.fit(np.radians(dataSelected))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([dataSelected[cluster_labels == n] for n in range(num_clusters)])

    tag = pd.DataFrame({
        "員編": group['員編'],
        "出勤日": group['出勤日'],
        "簽到退": group['簽到退'],
        "group": db.labels_
    })
    temp = pd.merge(group,tag, on=["員編","出勤日","簽到退"])
    
    output = pd.concat([output,temp],axis=0)

output.to_excel('C:\\Users\\vincentkuo\\Downloads\\signLocationByApp\\APP簽到地點_dbscan_epsilon_500m_minpts_test.xlsx',index=False)