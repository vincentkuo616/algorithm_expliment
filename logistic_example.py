# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:09:40 2020

@author: vincentkuo
"""


from sklearn import datasets
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix#, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
n_class = len(set(iris.target))                               # 类别数量

x, y = iris.data, iris.target
y_one_hot = label_binarize(y, np.arange(n_class))             # 转化为one-hot

# 建模
model = LogisticRegression()
model.fit(x, y)

# 预测值y的三种形式
y_score = model.predict(x)                                    # 形式一：原始值（0或1或2）
y_score_pro = model.predict_proba(x)                          # 形式二：各类概率值
y_score_one_hot = label_binarize(y_score, np.arange(n_class)) # 形式三：one-hot值

obj1 = confusion_matrix(y, y_score)
print('confusion_matrix\n', obj1)

print(model.decision_function(x))