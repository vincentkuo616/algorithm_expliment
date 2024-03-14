# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:55:22 2022

@author: vincentkuo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def readData(root):
    try:
        if root[-3:]=='csv':
            data = pd.read_csv(root,encoding='utf-8')
        elif root[-3:]=='lsx':
            data = pd.read_excel(root)
        elif root[-3:]=='txt':
            data = pd.read_csv(root,header=None,encoding='utf-8')
        else:
            data = pd.read_excel(root,encoding='utf-8')
    except:
        print("Data parsing failed")
    return data
'''
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
'''
prjRootFolder = "D:\\文件"
dataName = '\\4Kmeans(cmr).xlsx'
dataRoot = prjRootFolder+dataName
data = readData(dataRoot)
features = ['SALESAMT','OFFICEAMT','GGCNT','BrandCNT']
X = data[features]
normalized_X = (X-X.mean())/X.std()
normalized_X = normalized_X.values


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(normalized_X)

pca = PCA(n_components=1)
pca.fit(normalized_X)
X_pca = pca.transform(normalized_X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

X_new = pca.inverse_transform(X_pca)
#plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.7)
plt.axis('equal');