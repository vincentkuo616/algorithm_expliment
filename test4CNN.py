# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:34:22 2019

@author: vincentkuo
"""

from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Convolution1D, Activation, Conv1D, Embedding, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def show_train_history(train_history):
    import matplotlib.pyplot as plt
    flg = plt.gcf()
    flg.set_size_inches(16,6)
    plt.subplot(121)
    plt.plot(train_history.history['mae'])
    plt.plot(train_history.history['val_mae'])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel('mae')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.subplot(122)
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def fValue(y_true, y_pred):
    yy, yx, xy =0, 0, 0
    print(y_true)
    print(y_true[0])
    print(y_pred)
    print(y_pred[0])
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    for i in range(0,10143):
        if (y_true[i]==1) and (y_pred[i]==1):   yy+=1
        #if tf.cond(tf.equal(y_true[i],y_pred[i])) and tf.cond(tf.equal(y_true[i],1)) is not None:  yy+=1
        elif (y_true[i]==1) and (y_pred[i]==0):    yx+=1
        #elif tf.cond(tf.equal(y_pred[i]),0) and tf.cond(tf.equal(y_true[i],1)) is not None:  yx+=1
        elif (y_true[i]==0) and (y_pred[i]==1):    xy+=1
        #elif tf.cond(tf.equal(y_pred[i]),1) and tf.cond(tf.equal(y_true[i],0)) is not None:  xy+=1
    return yy*2/(yy*2+yx+xy)

def fValue2(y_true, y_pred):
    yy, yx, xy =0, 0, 0
    for i in range(0,10143):
        if (y_true[i]==1) and (y_pred[i]):   yy+=1
        #if tf.cond(tf.equal(y_true[i],y_pred[i])) and tf.cond(tf.equal(y_true[i],1)) is not None:  yy+=1
        elif (y_true[i]==1) and (not y_pred[i]):    yx+=1
        #elif tf.cond(tf.equal(y_pred[i]),0) and tf.cond(tf.equal(y_true[i],1)) is not None:  yx+=1
        elif (y_true[i]==0) and (y_pred[i]):    xy+=1
        #elif tf.cond(tf.equal(y_pred[i]),1) and tf.cond(tf.equal(y_true[i],0)) is not None:  xy+=1
    return yy*200/(yy*2+yx+xy)

prjRootFolder="C://Users//vincentkuo//Documents//vincent//Y191015//"
TrainSet = pd.read_csv(prjRootFolder+"TrainData_SM_ENN8%_Y191015.csv",encoding='utf-8')
#TrainSet = pd.read_csv(prjRootFolder+"TrainData_AllKNN_Y191015_new.csv",encoding='ISO-8859-15')
#TrainSet = pd.read_csv(prjRootFolder+"TrainData_AllKNNSMOTE_Y191015_new.csv",encoding='ISO-8859-15')
#TrainSet = pd.read_csv(prjRootFolder+"TrainDataOrderByIV_Y191015_new.csv",encoding='ISO-8859-15')
TestSet = pd.read_csv(prjRootFolder+"TestDataOrderByIV_Y191015_new.csv",encoding='ISO-8859-15')
x_train_data = TrainSet.iloc[:,5:69].values
y_train_label = TrainSet.iloc[:,4].values
x_test_data = TestSet.iloc[:,5:69].values
y_test_label = TestSet.iloc[:,4].values

np.random.seed(10)
#(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
print(x_train_data.shape)
print(x_test_data.shape)
print(y_train_label.shape)
print(y_test_label.shape)

x_train = x_train_data.reshape(x_train_data.shape[0],64,1).astype('float32')
x_test = x_test_data.reshape(10143,64,1).astype('float32')
x_train = x_train_data
x_test = x_test_data

x_train_mean = x_train.mean(axis=0)
x_test_mean = x_test.mean(axis=0)
x_train_std = x_train.std(axis=0)
x_test_std = x_test.std(axis=0)

x_train_normalize = (x_train-x_train_mean)/x_train_std
x_test_normalize = (x_test-x_test_mean)/x_test_std

#y_train_onehot = np_utils.to_categorical(y_train_label)
#y_test_onehot = np_utils.to_categorical(y_test_label)

def create_model():
    #create model
    model = Sequential()
    '''
    model.add(Embedding(10000000, 54, input_length=54))
    model.add(Conv1D(4,1,activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(4,1,activation='relu'))
    model.add(GlobalMaxPooling1D())
    '''
    #model.add(Conv1D(filters=1,
     #                filter_length=1,
      #               input_shape=(54,1)))
    #model.add(Convolution1D(filters=27,
     #                filter_length=1))
    model.add(Dense(109, kernel_initializer='uniform', input_dim=64))
    model.add(Activation('relu'))
    #model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(27))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(27))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    #model.add(Dense())
    model.add(Activation('sigmoid'))
    
    
    print(model.summary())
    
    sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='mae',optimizer=sgd,metrics=['accuracy'])
    
    #train model
    #model.compile(loss='categorical_crossentropy',
     #             optimizer='adam',metrics=['accuracy'])
    '''
    train_history=model.fit(x=x_train_normalize,
                            y=y_train_onehot,
                            validation_split=0.2,
                            epochs=13,
                            batch_size=1000, # 一個 epoch 要跑 （訓練樣本數/Batch_Size）個 Iterations
                            verbose=2)
    '''
    return model
#estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=50, verbose=2)
#kfold = StratifiedKFold(n_splits=3, shuffle=True)
#results = cross_val_score(estimator, x_train_normalize, y_train_label, cv=kfold)
#print(results.mean()*100, results.std()*100)
model = create_model()
model.fit(x_train_normalize, y_train_label, batch_size=50, epochs=200, verbose=0)
y_pred = model.predict(x_test_normalize)
y_pred3 = model.predict(x_train_normalize)
y_pred2 = (y_pred > 0.5)
print(confusion_matrix(y_test_label, y_pred2))
answer = model.predict_classes(x_test_normalize)
answer2 = model.predict_classes(x_train_normalize)
print(fValue2(y_test_label, y_pred2))

'''
print(train_history.history['mae'])

    
show_train_history(train_history)

scores = model.evaluate(x_test_normalize, y_test_onehot)
print('Accuracy=', scores[1])
prediction = model.predict_classes(x_test_normalize)
print(prediction)

print(pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict']))

df = pd.DataFrame({'label':y_test_label,'predict':prediction})
#print(df[(df.label==7) & (df.predict==2)])
'''