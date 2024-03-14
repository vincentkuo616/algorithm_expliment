import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.metrics import auc

y = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])

#y为数据的真实标签

scores = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.57,0.35,0.13,0.26,0.9,0.8,0.7,0.6,0.5,0.46,0.41,0.67,0.55,0.95])

#scores为分类其预测的得分

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

#得到fpr,tpr, thresholds

plt.plot(fpr,tpr,marker = 'o')

plt.show()

AUC = auc(fpr, tpr)

print(AUC)
