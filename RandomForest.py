# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:31:28 2019

@author: vincentkuo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier

def main():
    
    # import data C:\Users\vincentkuo\Documents\vincent\Y191206
    prjRootFolder="C://Users//vincentkuo//Documents//"
    dataset = pd.read_csv(prjRootFolder+"vincent//Y191206//RECE&COM_V2.8.csv",encoding='utf-8')
    dataset = pd.read_csv(prjRootFolder+"vincent//Y191206//arminvoice_forRF.csv",encoding='utf-16')
    #fromA = 18
    #toB = 27
    choose = [27,367,387,14,405,403,406,395,408,409,415,410,411,183,407,412,443]
    X = dataset.iloc[:,choose].values
    y = dataset.iloc[:,2].values
    fname = dataset.iloc[:,choose].columns.values
    
    
    # Build a forest and compute the feature importances
    forest = RandomForestClassifier(warm_start=True,
                                    random_state=0,
                                    max_depth=15,
                                    #bootstrap=True,
                                    #max_samples=0.2,
                                    min_samples_split=0.01,
                                    min_samples_leaf=5,
                                    max_leaf_nodes=100,
                                    oob_score=True)
    
    temp_model = [("RandomForestClassifier", forest)]
    
    error_rate = OrderedDict((label, []) for label, _ in temp_model)
    
    min_estimators = 1
    max_estimators = 200
    
    for label, clf in temp_model:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, y)
            
            # Record the OOB error for each 'n_estimators=i' setting.
            oob_error = 1 - forest.oob_score_
            error_rate[label].append((i, oob_error))
        
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)
    
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()
    
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(X.shape[1]):
        print(f,'.',fname[indices[f]],'(',importances[indices[f]],')')
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", align="center")
    plt.xticks(range(X.shape[1]), fname[indices], rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.show()

#try:
main()
#except:
#    print('--------------ERROR--------------')
