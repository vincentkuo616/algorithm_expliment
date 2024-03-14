# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:10:24 2022

@author: vincentkuo
"""

import numpy as np
import matplotlib.pyplot as plt

# setting the ranges for calculation
# MyVar: the higher the better detalization
MyVar = 100000

# 
startbee = -2
stopbee  = 2
bee = np.arange(startbee, stopbee, ((stopbee-startbee)/MyVar))

# 定義函數 Weierstrass function
def weierstrass(x,Nvar):
    we=np.zeros(MyVar)
    for n in range(0,Nvar):
        we=we+np.cos(3**n*np.pi*x)/2**n
    return we

# 畫圖時，x 跟 y 軸的 ranges相同
plt.plot(bee,np.reshape(weierstrass(bee, 500),(MyVar,)))

plt.show()