# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:06:55 2019

@author: vincentkuo
"""
import scipy
from  scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans as k_means
from sklearn.metrics import davies_bouldin_score,calinski_harabasz_score,silhouette_score
#adjusted_rand_score,adjusted_mutual_info_score,v_measure_score,fowlkes_mallows_score 這些都要有label
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from jqm_cvi.jqmcvi import base  # https://github.com/jqmviegas/jqm_cvi 下載壓縮檔後 解壓縮丟到site-package 把資料夾改名成jqm_cvi

import warnings
warnings.filterwarnings('ignore')

dst = euclidean
 
k_means_args_dict = {
    'n_clusters': 0,
    # drastically saves convergence time
    'init': 'k-means++',
    'max_iter': 100,
    'n_init': 1,
    'verbose': False,
    # 'n_jobs':8
}
 
 
def gap(data, refs=None, nrefs=20, ks=range(1, 8)):
    """
    I: NumPy array, reference matrix, number of reference boxes, number of clusters to test
    O: Gaps NumPy array, Ks input list
 
    Give the list of k-values for which you want to compute the statistic in ks. By Gap Statistic
    from Tibshirani, Walther.
    """
    shape = data.shape
 
    if not refs:
        tops = data.max(axis=0)
        bottoms = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops - bottoms))
        rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i] * dists + bottoms
    else:
        rands = refs
 
    gaps = scipy.zeros((len(ks),))
 
    for (i, k) in enumerate(ks):
        k_means_args_dict['n_clusters'] = k
        kmeans = k_means(**k_means_args_dict)
        kmeans.fit(data)
        (cluster_centers, point_labels) = kmeans.cluster_centers_, kmeans.labels_
 
        disp = sum(
            [dst(data[current_row_index, :], cluster_centers[point_labels[current_row_index], :]) for current_row_index
             in range(shape[0])])
 
        refdisps = scipy.zeros((rands.shape[2],))
 
        for j in range(rands.shape[2]):
            kmeans = k_means(**k_means_args_dict)
            kmeans.fit(rands[:, :, j])
            (cluster_centers, point_labels) = kmeans.cluster_centers_, kmeans.labels_
            refdisps[j] = sum(
                [dst(rands[current_row_index, :, j], cluster_centers[point_labels[current_row_index], :]) for
                 current_row_index in range(shape[0])])
 
        # let k be the index of the array 'gaps'
        gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)
 
    return ks, gaps





# 3 points in dataset
data = np.array([[1,8],[2,6],[3,7],
                 [6,2],[7,5],[8,2]
                 ,[10,3],[9,5],[11,4]
                 ])
#AllData = pd.read_csv(r"C:\Users\vincentkuo\Documents\vincent\Y191101\TakeALookData2_Cluster_STD.csv",encoding='ISO-8859-15')
#data = AllData.iloc[:,116:119].values

AllData = pd.read_csv(r"D:\文件\vincent\Y191101\TakeALookData3_test1.csv",encoding='ISO-8859-15')
data = AllData.iloc[:,6:8].values

AllData = pd.read_excel(r"D:\文件\4Kmeans(cmr)_kmeansStatistica.xlsx")
data = AllData.iloc[:,[1,3,4,5,6,7]]
data = (data-data.mean()) / data.std()
data = data.values
'''
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
labels = kmeans.predict(data)
print(kmeans.predict(data))
print(kmeans.cluster_centers_)
print(davies_bouldin_score(data, labels)  )

plt.scatter(data[:,0],data[:,1])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='red')
plt.show()
'''
davies_bouldin_arr = []
calinski_harabasz_arr=[]
silhouette_arr=[]
dunn_index_arr=[]
gap_statistic_arr=[]
arr=[]
for i in range(2,11):
    #kmeans2=KMeans(n_clusters=i,random_state=123).fit(data)
    label = AllData.iloc[:,6+i]
    print(label)
    print('========',i,'=====')
    arr.append(i)
    #davies_bouldin
    d_score = davies_bouldin_score(data, label)
    davies_bouldin_arr.append(d_score)
    print('聚類%d簇的DBI分數為：%f'%(i,d_score))#越低越好
    #calinski_harabasz
    score=calinski_harabasz_score(data,label)
    calinski_harabasz_arr.append(score)
    print('聚類%d簇的calinski_harabaz分數為：%f'%(i,score)) #越高越好
    # silhouette
    s_score =silhouette_score(data, label)
    silhouette_arr.append(s_score)
    print('聚類%d簇的silhouette分數為：%f'%(i,d_score))#越高越好
    #dunn index
    k_list = [data, label]
    #dunn_score = base.dunn(k_list)
    #dunn_index_arr.append(dunn_score)
    #print('聚類%d簇的dunn分數為：%f'%(i,dunn_score))#越高越好
    #gap statistic
    #ks,gaps = gap(data,ks=range(i-1, i))
    #gap_statistic_arr.append(gaps[0])
    #print('聚類%d簇的gaps分數為：%f'%(i,gaps[0]))#越高越好



# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離
plt.title("davies_bouldin , be lower be better", x=0.5, y=1.03)
# 把資料放進來並指定對應的X軸、Y軸的資料，用方形做標記(s-)，並指定線條顏色為紅色，使用label標記線條含意
plt.xticks(range(2,11))
plt.plot(arr,davies_bouldin_arr,'s-',color = 'r', label="davies_bouldin")
plt.show()
plt.title("calinski_harabasz , be higher be better", x=0.5, y=1.03)
plt.xticks(range(2,11))
plt.plot(arr,calinski_harabasz_arr,'s-',color = 'green', label="calinski_harabasz")
plt.show()
plt.title("silhouette, be higher be better", x=0.5, y=1.03)
plt.xticks(range(2,11))
plt.plot(arr,silhouette_arr,'s-',color = 'blue', label="silhouette")
plt.show()
#plt.title("dunn_index, be higher be better", x=0.5, y=1.03)
#plt.plot(arr,dunn_index_arr,'s-',color = 'grey', label="dunn_index")
#plt.show()
#plt.title("gap_statistic, be higher be better", x=0.5, y=1.03)
#plt.plot(arr,gap_statistic_arr,'s-',color = 'black', label="gap_statistic")
#plt.show()