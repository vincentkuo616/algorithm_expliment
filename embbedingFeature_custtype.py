# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:17:29 2021

@author: vincentkuo
"""

import pandas as pd
import categorical_embedder as ce 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C://Users//vincentkuo//Downloads//Custtype_Data_2.csv', encoding='utf-8') 
X = df.drop(['CUSTOMERCODE', 'CSG','PS'], axis=1)
# X = df[{'CUSTTYPE','AMTperM_BIN'}]
y = df['CSG']

# prepare target
def prepare_targets(y_train):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	return y_train_enc

y = pd.Series(prepare_targets(y))

embedding_info = ce.get_embedding_info(X) 
X_encoded,encoders = ce.get_label_encoded_data(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded,y)

embedding_4 = ce.get_embeddings(X_train, y_train, categorical_embedding_info=embedding_info, 
                            is_classification=True, epochs=100,batch_size=256)

dfs = ce.get_embeddings_in_dataframe(embeddings=embedding_4, encoders=encoders)

print(dfs['AMTperM_BIN'])

print(dfs['CUSTTYPE'])

data_4 = ce.fit_transform(X, embeddings=embedding_4, encoders=encoders, drop_categorical_vars=True)

print(data_4.head())