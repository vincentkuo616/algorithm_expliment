# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:48:00 2020

@author: vincentkuo
"""


import numpy as np
import pandas as pd 

data = {'score': [91, 93, 72, 87, 86, 73, 68, 87, 78, 99, 95, 76, 84, 96, 76, 80, 83, 84, 73, 74],
        'hours': [16, 6, 3, 1, 2, 3, 2, 5, 2, 5, 2, 3, 4, 3, 3, 3, 4, 3, 4, 4],
        'prep': [3, 4, 0, 3, 4, 0, 1, 2, 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2],
        'grade': [70, 88, 80, 83, 88, 84, 78, 94, 90, 93, 89, 82, 95, 94, 81, 93, 93, 90, 89, 89]
        }
data = {'score': [91, 93, 72, 87, 86, 73, 68, 87, 78, 99, 95, 76, 84, 96, 76, 80, 83, 84, 73, 74, 82.75],
        'hours': [16, 6, 3, 1, 2, 3, 2, 5, 2, 5, 2, 3, 4, 3, 3, 3, 4, 3, 4, 4, 3.9],
        'prep': [3, 4, 0, 3, 4, 0, 1, 2, 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2, 2.25],
        'grade': [70, 88, 80, 83, 88, 84, 78, 94, 90, 93, 89, 82, 95, 94, 81, 93, 93, 90, 89, 89, 87.15]
        }
data = {'score': [91, 93, 72, 87, 86],
        'hours': [16, 6, 3, 1, 2],
        'prep': [3, 4, 0, 3, 4],
        'grade': [70, 88, 80, 83, 88]
        }
'''
data = {'score': [91, 93, 72, 87, 86, 85.8],
        'hours': [16, 6, 3, 1, 2, 5.6],
        'prep': [3, 4, 0, 3, 4, 2.8],
        'grade': [70, 88, 80, 83, 88, 81.8]
        }
data = {'score': [0, 3, 12],
        'hours': [2, 0, 8]
        }
data = {'score': [91, 93, 72, 87, 86, 85.8, 85.8],
        'hours': [16, 6, 3, 1, 2, 5.6, 5.6],
        'prep': [3, 4, 0, 3, 4, 2.8, 2.8],
        'grade': [70, 88, 80, 83, 88, 81.8, 81.8]
        }
data = {'score': [0.631988, 0.875061, -1.6772, 0.145843, 0.024307],
        'hours': [1.70286, 0.065495, -0.42571, -0.75319, -0.58945],
        'prep': [0.121716, 0.730297, -1.70403, 0.121716, 0.730297],
        'grade': [-1.58823, 0.834492, -0.24227, 0.161515, 0.834492]
        }
'''

df = pd.DataFrame(data,columns=['score', 'hours', 'prep','grade'])
#df = pd.DataFrame(data,columns=['score', 'hours'])
print(df.head())
df_normal = (df-df.mean())/df.std()
print(df.mean())

#create function to calculate Mahalanobis distance
def mahalanobis(x=None, data=None, cov=None):

    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()

#create new column in dataframe that contains Mahalanobis distance for each row
df['mahalanobis'] = mahalanobis(x=df, data=df[['score', 'hours', 'prep', 'grade']])
df_normal['mahalanobis'] = mahalanobis(x=df_normal, data=df_normal[['score', 'hours', 'prep', 'grade']])
#df['mahalanobis'] = mahalanobis(x=df, data=df[['score', 'hours']])
#df_normal['mahalanobis'] = mahalanobis(x=df_normal, data=df_normal[['score', 'hours']])

#display first five rows of dataframe
print(df)
print(df_normal.head())

from sklearn.covariance import EmpiricalCovariance, MinCovDet

#不使用估計
print(EmpiricalCovariance(df.values))
print(MinCovDet(df.values))

#使用估計
MLE = EmpiricalCovariance().fit(df.values)
print(MLE.mahalanobis(df.values))
MCD = MinCovDet().fit(df.values)
print(MCD.mahalanobis(df.values))
