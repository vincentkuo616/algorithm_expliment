# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:21:41 2022

@author: syn009
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.simplefilter("ignore")


prjRootFolder="C:\\WorkingArea\\Vincent\\DBS\\Y2211-業績分群\\"
readfile="Apriori_result"
df_name = 'CSG'


df_all = pd.read_table(r"C:\\WorkingArea\\Vincent\\DBS\\Y2211-業績分群\\4Apriori.txt",encoding='utf-16')

df = df_all['CSG']
df_name = 'CSG'

dataset = []
tmp_arr = np.array(df)
for i in range(0, len(tmp_arr)):
    tmp = tmp_arr[i][0].replace(' ','') .split(',')
    dataset.append(tmp)

 # apriori function 要求 data 使用 pandas DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print('開始計算',df_name)
#用 Apriori 找出 frequent itemsets
#最低支持度 (min support) 定為 60%，即在所有交易中，該產品最少佔 60%，才可符合 frequent itemset 的要求。
#print(apriori(df, min_support=0.0001, use_colnames=True))
#加入ITEM個數
#0.0002
frequent_itemsets = apriori(df, min_support=0.0005, use_colnames=True, max_len=2)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
print('計算單一support完成，準備存檔')
frequent_itemsets.to_csv(prjRootFolder+readfile+"_"+df_name +"_Y2009_single_all.csv",sep=",",index=False,encoding='utf-16')
#從 frequent itemsets 中 generate association rules
#下例是 找出confidence>0.7的

rules =  association_rules(frequent_itemsets,metric="lift",min_threshold=1.2)
print(rules)
'''
若要找出lift >1.2的 如下:
rules =  association_rules(frequent_itemsets,metric="lift",min_threshold=1.2)
'''
#顯示 antecedent 的長度
rules["antcedent_len"] = rules["antecedents"].apply(lambda x: len(x))
print('計算完成，準備存檔')

rules.to_csv(prjRootFolder+readfile+"_"+df_name +"_byCSG_all.csv",sep=",",index=False,encoding='utf-16')