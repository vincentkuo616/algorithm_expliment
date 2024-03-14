# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:01:38 2021

@author: vincentkuo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# define samples this way as scipy.stats.wasserstein_distance can't take probability distributions directly
sampP = [1,1,1,1,1,1,2,3,4,5]
sampQ = [1,2,3,4,5,5,5,5,5,5]
#sampP = [1,2,2,3,3,3,3,4,4,4]
#sampQ = [1,1,2,3,3,4,4,4,4,4]
sampP = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]
sampQ = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]

#sampP = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]
#sampQ = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

# and for scipy.stats.entropy (gives KL divergence here) we want distributions
P = np.unique(sampP, return_counts=True)[1] / len(sampP)
Q = np.unique(sampQ, return_counts=True)[1] / len(sampQ)
# compare to this sample / distribution:
sampQ2 = [1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]
Q2 = np.unique(sampQ2, return_counts=True)[1] / len(sampQ2)

P = np.array([0.1,0.1,0.2,0.2,0.1,0.3])
Q = np.array([0.2,0.1,0.2,0.15,0.05,0.3])
Q2 = np.array([0.05,0.1,0.15,0.2,0.2,0.3])

fig = plt.figure(figsize=(12,9))
fig.subplots_adjust(wspace=0.5)
plt.subplot(2,2,1)
plt.bar(np.arange(len(P)), P, color='r')
plt.xticks(np.arange(len(P)), np.arange(1,7), fontsize=0)
plt.subplot(2,2,3)
plt.bar(np.arange(len(Q)), Q, color='b')
plt.xticks(np.arange(len(Q)), np.arange(1,7))
plt.title("Wasserstein distance {:.4}\nKL divergence {:.4}\nJS divergence {:.4}".format(
    scipy.stats.wasserstein_distance(sampP, sampQ), scipy.stats.entropy(P, Q), JS_divergence(P, Q)), fontsize=10)
plt.subplot(2,2,2)
plt.bar(np.arange(len(P)), P, color='r')
plt.xticks(np.arange(len(P)), np.arange(1,7), fontsize=0)
plt.subplot(2,2,4)
plt.bar(np.arange(len(Q2)), Q2, color='b')
plt.xticks(np.arange(len(Q2)), np.arange(1,7))
plt.title("Wasserstein distance {:.4}\nKL divergence {:.4}\nJS divergence {:.4}".format(
    scipy.stats.wasserstein_distance(sampP, sampQ2), scipy.stats.entropy(P, Q2), JS_divergence(P, Q2)), fontsize=10)
plt.show()