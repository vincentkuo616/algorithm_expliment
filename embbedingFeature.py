# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:34:10 2021

@author: vincentkuo
"""

import numpy as np

import tensorflow as tf

from keras.models import Model

from keras.layers import Input, Dense, Concatenate, Reshape, Dropout

from keras.layers.embeddings import Embedding

from tensorflow.python.keras import backend as K

from keras.utils import plot_model

tf.compat.v1.disable_v2_behavior()

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

K.set_session(sess)

 

 

def build_embedding_network():

    """

    以網路結構embeddding層在前，dense層在後。即訓練集的X必須以分類特徵在前，連續特徵在後。

    """

    inputs = []

    embeddings = []

 

    input_cate_feature_1 = Input(shape=(1,))

    embedding = Embedding(10, 3, input_length=1)(input_cate_feature_1)

    embedding = Reshape(target_shape=(3,))(embedding)  # embedding後是10*1*3，為了後續計算方便，因此使用Reshape轉為10*3

    inputs.append(input_cate_feature_1)

    embeddings.append(embedding)

 

    input_cate_feature_2 = Input(shape=(1,))

    embedding = Embedding(4, 2, input_length=1)(input_cate_feature_2)

    embedding = Reshape(target_shape=(2,))(embedding)

    inputs.append(input_cate_feature_2)

    embeddings.append(embedding)

 

    input_numeric = Input(shape=(1,))

    embedding_numeric = Dense(16)(input_numeric)

    inputs.append(input_numeric)

    embeddings.append(embedding_numeric)

 

    x = Concatenate()(embeddings)

    x = Dense(10, activation='relu')(x)

    x = Dropout(.15)(x)

    output = Dense(1, activation='sigmoid')(x)

 

    model = Model(inputs, output)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

 

 

"""

構造訓練數據

輸入資料是320*3，320個樣本，2個類別特徵，1個連續特徵。

對類別特徵做entity embedding，第一個類別特徵10個，第二個類別特徵4個。對這2個特徵做one-hot的話，應該為10+4，

對第一個類別特徵做embedding使其為3維，對第二個類別特徵做embedding使其為2維。3對連續特徵不做處理。這樣理想輸出的結果就應該是3+2+1。

維和2維的設定是根據實驗效果和交叉驗證設定。

"""

sample_num = 320  # 樣本數為320

cate_feature_num = 2  # 類別特徵為2

contious_feature_num = 1  # 連續特徵為1

 

rng = np.random.RandomState(0)  # 保證了訓練集的複現

cate_feature_1 = rng.randint(10, size=(sample_num, 1))

cate_feature_2 = rng.randint(4, size=(sample_num, 1))

contious_feature = rng.rand(sample_num, 1)

 

X = [cate_feature_1, cate_feature_2, contious_feature]

Y = np.random.randint(2, size=(sample_num, 1))  # 二分類

 

cate_embedding_dimension = {'0': 3, '1': 2}  # 記錄類別特徵embedding後的維度。key為類別特徵索引，value為embedding後的維度

 

"""

訓練和預測

"""

NN = build_embedding_network()

# plot_model(NN, to_file='NN.png')  # 畫出模型，需要GraphViz包。另外需要安裝 pip install pydot

 

NN.fit(X, Y, epochs=3, batch_size=4, verbose=0)

y_preds = NN.predict(X)[:, 0]

 

"""

讀embedding層的輸出結果

"""

model = NN  # 創建原始模型

for i in range(cate_feature_num):

    layer_name = NN.get_config()['layers'][cate_feature_num + i]['name']  # cate_feature_num+i就是所有embedding層

    intermediate_layer_model = Model(inputs=NN.input, outputs=model.get_layer(layer_name).output)

    intermediate_output = intermediate_layer_model.predict(X)

    intermediate_output.resize([sample_num, cate_embedding_dimension[str(i)]])

    if i == 0:

        X_embedding_trans = intermediate_output

    else:

        X_embedding_trans = np.hstack((X_embedding_trans, intermediate_output))  # 水準拼接

 

for i in range(contious_feature_num):

    if i == 0:

        X_contious = X[cate_feature_num + i]

    else:

        X_contious = np.hstack((X_contious, X[cate_feature_num + i]))

 

X_trans = np.hstack((X_embedding_trans, X_contious))  # 在類別特徵做embedding後的基礎上，拼接連續特徵，形成最終矩陣，也就是其它學習器的輸入。

 

print(X_trans[:5])  # 其中，類別特徵維度為5（前5個），連續特徵維度為1（最後1個）

 

weight = NN.trainable_weights[0].eval(session=sess) # embedding_1層的參數。

print(weight[:5])
