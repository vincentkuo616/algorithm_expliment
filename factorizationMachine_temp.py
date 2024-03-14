# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:57:10 2021

@author: vincentkuo
"""


from itertools import count # 迭代器
from collections import defaultdict # 使用dict時，如果引用的Key不存在，就會拋出KeyError。如果希望key不存在時，返回一個默認值，就可以用defaultdict
from scipy.sparse import csr # csr_matrix，全名爲Compressed Sparse Row，是按行對矩陣進行壓縮的。CSR需要三類數據：數值，列號，以及行偏移量。CSR是一種編碼的方式，其中，數值與列號的含義，與coo裏是一致的。行偏移表示某一行的第一個元素在values裏面的起始偏移位置。 
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm # 可以顯示循環的進度條的庫
tf.compat.v1.disable_v2_behavior()

def vectorize_dic(dic, ix=None, p=None, n=0, g=0):
    """ 
    Creates a scipy csr matrix from a list of lists (each inner list is a set of values corresponding to a feature) 
    
    parameters:
    -----------
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of featrure space (number of columns in the sparse matrix) (default None)
    """
    if (ix == None):
        d = count(0)
        ix = defaultdict(lambda: next(d)) 
        
    n = len(list(dic.values())[0]) # num samples
    g = len(list(dic.keys())) # num groups
    nz = n * g # number of non-zeros

    col_ix = np.empty(nz, dtype=int)     
    
    i = 0
    for k, lis in dic.items():     
        # append index el with k in order to prevet mapping different columns with same id to same index
        col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
        i += 1
        
    row_ix = np.repeat(np.arange(0, n), g)      
    data = np.ones(nz)
    
    if (p == None):
        p = len(ix)
        
    ixx = np.where(col_ix < p)

    return csr.csr_matrix((data[ixx],(row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix

cols = ['user','item','rating','timestamp']

train = pd.read_csv('C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\ratings_100.csv',delimiter='\t',names = cols)
test = pd.read_csv('C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\ratings_100.csv',delimiter='\t',names = cols)

x_train,ix = vectorize_dic({'users':train['user'].values,  'items':train['item'].values},n=len(train.index),g=2)

x_test,ix = vectorize_dic({'users':test['user'].values,   'items':test['item'].values},ix,x_train.shape[1],n=len(test.index),g=2)

print(x_train)
y_train = train['rating'].values
y_test = test['rating'].values

x_train = x_train.todense() # toarray returns an ndarray; todense returns a matrix. If you want a matrix, use todense otherwise, use toarray
x_test = x_test.todense()

#%% 生成器
def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

#%% 估計值計算
n,p = x_train.shape

k = 10

x = tf.placeholder('float',[None,p])

y = tf.placeholder('float',[None,1])

w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([p]))

v = tf.Variable(tf.random_normal([k,p],mean=0,stddev=0.01))

#y_hat = tf.Variable(tf.zeros([n,1]))

linear_terms = tf.add(w0,tf.reduce_sum(tf.multiply(w,x),1,keep_dims=True)) # n * 1
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(
            tf.matmul(x,tf.transpose(v)),2),
        tf.matmul(tf.pow(x,2),tf.transpose(tf.pow(v,2)))
    ),axis = 1 , keep_dims=True)

y_hat = tf.add(linear_terms,pair_interactions)

#%% 損失函數計算
lambda_w = tf.constant(0.001,name='lambda_w')
lambda_v = tf.constant(0.001,name='lambda_v')

l2_norm = tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w,tf.pow(w,2)),
        tf.multiply(lambda_v,tf.pow(v,2))
    )
)

error = tf.reduce_mean(tf.square(y-y_hat))
loss = tf.add(error,l2_norm)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

#%% 模型訓練
epochs = 10
batch_size = 1000

# Launch the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(x_train.shape[0]) # 函數shuffle與permutation都是對原來的數組進行重新洗牌（即隨機打亂原來的元素順序）；區別在於shuffle直接在原來的數組上進行操作，改變原來數組的順序，無返回值。而permutation不直接在原來的數組上進行操作，而是返回一個新的打亂順序的數組，並不改變原來的數組。
        # iterate over batches
        for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
            _,t = sess.run([train_op,loss], feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
            print(t)


    errors = []
    for bX, bY in batcher(x_test, y_test):
        errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
        print(errors)
    RMSE = np.sqrt(np.array(errors).mean())
    print (RMSE)