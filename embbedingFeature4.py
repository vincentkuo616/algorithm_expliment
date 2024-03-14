# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:35:41 2021

@author: vincentkuo
"""

import pandas as pd
import categorical_embedder as ce 
from sklearn.model_selection import train_test_split


df = pd.read_csv('C://Users//vincentkuo//Downloads//HR_Attrition_Data.csv') 
X = df.drop(['employee_id', 'is_promoted'], axis=1) 
y = df['is_promoted']

embedding_info = ce.get_embedding_info(X) 
X_encoded,encoders = ce.get_label_encoded_data(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded,y)

embedding_4 = ce.get_embeddings(X_train, y_train, categorical_embedding_info=embedding_info, 
                            is_classification=True, epochs=100,batch_size=256)

dfs = ce.get_embeddings_in_dataframe(embeddings=embedding_4, encoders=encoders)

print(dfs['department'])

print(dfs['education'])

data_4 = ce.fit_transform(X, embeddings=embedding_4, encoders=encoders, drop_categorical_vars=True)

print(data_4.head())