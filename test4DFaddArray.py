# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:14:21 2022

@author: vincentkuo
"""


import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(info+'...\n\n')
    
    
#================================================================================
# 一，讀取資料
#================================================================================
printlog("step1: reading data...")

# 讀取dftrain,dftest
breast = datasets.load_breast_cancer()
df = pd.DataFrame(breast.data,columns = [x.replace(' ','_') for x in breast.feature_names])
df['label'] = breast.target

tempList=[]
for i in range(569):
    tempList.append([np.random.randn(),np.random.randn()])
tempArray = np.array(tempList)
print(tempArray)

df.insert(df.shape[1], 'Embedding', 0)
df['Embedding'] = tempArray

print(df)

# dftrain,dftest = train_test_split(df)

# xgb_train = xgb.DMatrix(dftrain.drop("label",axis = 1),dftrain[["label"]])
# xgb_valid = xgb.DMatrix(dftest.drop("label",axis = 1),dftest[["label"]])

# #================================================================================
# # 二，設定引數
# #================================================================================
# printlog("step2: setting parameters...")
                               
# num_boost_round = 100                   
# early_stopping_rounds = 20

# # 配置xgboost模型引數
# params_dict = dict()

# # booster引數
# params_dict['learning_rate'] = 0.05      # 學習率，通常越小越好。
# params_dict['objective'] = 'binary:logistic'

# # tree引數
# params_dict['max_depth'] = 3              # 樹的深度，通常取值在[3,10]之間
# params_dict['min_child_weight']= 30       # 最小葉子節點樣本權重和，越大模型越保守。
# params_dict['gamma']= 0                   # 節點分裂所需的最小損失函式下降值，越大模型越保守。
# params_dict['subsample']= 0.8             # 橫向取樣，樣本取樣比例，通常取值在 [0.5，1]之間 
# params_dict['colsample_bytree'] = 1.0     # 縱向取樣，特徵取樣比例，通常取值在 [0.5，1]之間 
# params_dict['tree_method'] = 'hist'       # 構建樹的策略,可以是auto, exact, approx, hist

# # regulazation引數 
# # Omega(f) = gamma*T + reg_alpha* sum(abs(wj)) + reg_lambda* sum(wj**2)  
# params_dict['reg_alpha'] = 0.0            #L1 正則化項的權重係數，越大模型越保守，通常取值在[0,1]之間。
# params_dict['reg_lambda'] = 1.0           #L2 正則化項的權重係數，越大模型越保守，通常取值在[1,100]之間。

# # 其他引數
# params_dict['eval_metric'] = 'auc'
# params_dict['silent'] = 1
# params_dict['nthread'] = 2
# params_dict['scale_pos_weight'] = 1       #不平衡樣本時設定為正值可以使演算法更快收斂。
# params_dict['seed'] = 0


# #================================================================================
# # 三，訓練模型
# #================================================================================
# printlog("step3: training model...")


# result = {}
# watchlist = [(xgb_train, 'train'),(xgb_valid,'valid')] 

# bst = xgb.train(params = params_dict, dtrain = xgb_train, 
#                 num_boost_round = num_boost_round, 
#                 verbose_eval= 1,
#                 evals = watchlist,
#                 early_stopping_rounds=early_stopping_rounds,
#                 evals_result = result)


# #================================================================================
# # 四，評估模型
# #================================================================================
# printlog("step4: evaluating model ...")


# y_pred_train = bst.predict(xgb_train, ntree_limit=bst.best_iteration)
# y_pred_test = bst.predict(xgb_valid, ntree_limit=bst.best_iteration)

# print('train accuracy: {:.5} '.format(accuracy_score(dftrain['label'], y_pred_train>0.5)))
# print('valid accuracy: {:.5} \n'.format(accuracy_score(dftest['label'], y_pred_test>0.5)))

# # %matplotlib inline
# # %config InlineBackend.figure_format = 'svg'

# dfresult = pd.DataFrame({(dataset+'_'+feval): result[dataset][feval] 
#                for dataset in ["train","valid"] for feval in ['auc']})

# dfresult.index = range(1,len(dfresult)+1) 
# ax = dfresult.plot(kind='line',figsize=(8,6),fontsize = 12,grid = True) 
# ax.set_title("Metric During Training",fontsize = 12)
# ax.set_xlabel("Iterations",fontsize = 12)
# ax.set_ylabel("auc",fontsize = 12)


# ax = xgb.plot_importance(bst,importance_type = "gain",xlabel='Feature Gain')
# ax.set_xlabel("Feature Gain",fontsize = 12)
# ax.set_ylabel("Features",fontsize = 12)
# fig = ax.get_figure() 
# fig.set_figwidth(8)
# fig.set_figheight(6)

# #================================================================================
# # 五，儲存模型
# #================================================================================
# # printlog("step5: saving model ...")
# # model_dir = "data/bst.model"
# # print("model_dir: %s"%model_dir)
# # bst.save_model(model_dir)
# # bst_loaded = xgb.Booster(model_file=model_dir)

# printlog("task end...")