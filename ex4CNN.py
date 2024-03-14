# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:34:22 2019

@author: vincentkuo
"""

from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def show_train_history(train_history):
    import matplotlib.pyplot as plt
    flg = plt.gcf()
    flg.set_size_inches(16,6)
    plt.subplot(121)
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.subplot(122)
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

np.random.seed(10)
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
print(x_train_image.shape)
print(x_test_image.shape)
print(y_train_label.shape)
print(y_test_label.shape)

x_train = x_train_image.reshape(60000,28,28,1).astype('float32')
x_test = x_test_image.reshape(10000,28,28,1).astype('float32')

x_train_normalize = x_train/255
x_test_normalize = x_test/255

y_train_onehot = np_utils.to_categorical(y_train_label)
y_test_onehot = np_utils.to_categorical(y_test_label)

#create model
model = Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

print(model.summary())

#train model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_train_normalize,
                        y=y_train_onehot,
                        validation_split=0.2,
                        epochs=10,
                        batch_size=300,
                        verbose=2)

print(train_history.history['accuracy'])

    
show_train_history(train_history)

scores = model.evaluate(x_test_normalize, y_test_onehot)
print('Accuracy=', scores[1])
prediction = model.predict_classes(x_test_normalize)
print(prediction)

import pandas as pd
print(pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict']))

df = pd.DataFrame({'label':y_test_label,'predict':prediction})
print(df[(df.label==7) & (df.predict==2)])