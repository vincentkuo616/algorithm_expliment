# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:38:26 2020

@author: vincentkuo
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv

c=['#415952', '#f35134', '#243AB5']
def dataset_ (n=200, idx_outlier=0, ydistance=5):
    rng = np.random.RandomState(4)
    data  = np.dot(rng.rand(2, 2), rng.randn(2, n)).T
    data[idx_outlier:idx_outlier+1,1] = ydistance
    return data

N=10000
inx=5
i = dataset_(N, inx)
carray=[c[0]]*N
carray=np.full(N,c[0])
carray[inx]='blue'
print(carray[inx+1])
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(i[:,0], i[:,1], c=carray, alpha=0.3)

f_H = lambda i: np.dot(i, inv(np.dot(i.T, i))).dot(i.T)
H = f_H(i)
h=np.trace(H)/N # if h2* > 1 then not cutoff is applied

outliers=[]
for row in range(N):
    for col in range(N):
        if row == col:
            if H[row, col] > 2.*h:
                outliers.append(row)
     
carray=[c[0]]*N
carray = np.array(carray)
carray[outliers]='red'
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(i[:,0], i[:,1], c=carray, alpha=0.3)