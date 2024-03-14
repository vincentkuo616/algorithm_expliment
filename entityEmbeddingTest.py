# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:05:39 2021

@author: vincentkuo
"""


import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import tensorflow as tf
import random as rn

# ================================================
# 保證結果的復現

# import os
# os.environ['PYTHONHASHSEED'] = '0'

# np.random.seed(42)

# rn.seed(12345)

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# from keras import backend as K

# tf.set_random_seed(1234)

# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# ================================================

'''
输入数据是32*2，32个样本，2个类别特征，且类别特征的可能值是0到9之间（10个）。
对这2个特征做one-hot的话，应该为32*20，
embedding就是使1个特征原本应该one-hot的10维变为3维（手动设定，也可以是其它），因为有2个类别特征
这样输出的结果就应该是32*6
'''
model = Sequential()
model.add(Embedding(10, 3, input_length=2))
# model.add(Embedding(7, 2, input_length=5))
 
# 构造输入数据
input_array = np.random.randint(10, size=(32, 2))
# input_array = np.array([[5,1,2,3,6]])
 
# 搭建模型
model.compile('rmsprop', 'mse')
 
# 得到输出数据 输出格式为32*2*3。我们最终想要的格式为32*6，其实就是把2*3按照行拉成6维，然后就是我们对类别特征进行
# embedding后得到的结果了。
output_array = model.predict(input_array)
 
# 查看权重参数
weight = model.get_weights()
 
'''
我们肯定好奇：output_array是怎么得到的？
我们先来看weight的内容：10*3。这是什么意思呢，就是其实就是一个索引的结果表，如果原来特征值为0，那么就找第一行，如果原来特征值为3，
那么就找第4行。
0.00312117  -0.0475833  0.0386381
0.0153809   -0.0185934  0.0234457
0.0137821   0.00433551  0.018144
0.0468446   -0.00687895 0.0320682
0.0313594   -0.0179525  0.03054
0.00135239  0.0309016   0.0453686
0.0145149   -0.0165581  -0.0280098
0.0370018   -0.0200525  -0.0332663
0.0330335   0.0110769   0.00161555
0.00262188  -0.0495747  -0.0343777
以input_array的第一行为例
input_array的第一行是7和4，那么就找第8行和第5行，形成了output_array的第一个2*3，即
0.0370018   -0.0200525  -0.0332663
0.0313594   -0.0179525  0.03054
然后，拉成一个向量0.0370018  -0.0200525  -0.0332663 0.0313594    -0.0179525  0.03054
这就是原始特征值8和5经过embedding层后的转换结果!
'''