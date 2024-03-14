# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:05:20 2021

@author: vincentkuo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras.models import Sequential
from keras.utils import plot_model

def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = pd.read_csv("C://Users//vincentkuo//Downloads//"+ filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	# format all fields as string
	X = X.astype(str)
	# reshape target to be a 2d array
	y = y.reshape((len(y), 1))
	return X, y



# prepare input data OrdinalEncoder
def prepare_inputs_oe(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

# prepare input data OneHotEncoder
def prepare_inputs_ohe(X_train, X_test):
	ohe = OneHotEncoder()
	ohe.fit(X_train)
	X_train_enc = ohe.transform(X_train)
	X_test_enc = ohe.transform(X_test)
	return X_train_enc, X_test_enc

# prepare input data LabelEncoder
def prepare_inputs_le(X_train, X_test):
	X_train_enc, X_test_enc = list(), list()
	# label encode each column
	for i in range(X_train.shape[1]):
		le = LabelEncoder()
		le.fit(X_train[:, i])
		# encode
		train_enc = le.transform(X_train[:, i])
		test_enc = le.transform(X_test[:, i])
		# store
		X_train_enc.append(train_enc)
		X_test_enc.append(test_enc)
	return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

embedLabel = True


# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

if embedLabel:
    print('Embedding')
    # prepare input data
    X_train_enc, X_test_enc = prepare_inputs_le(X_train, X_test)
    # prepare output data
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    # make output 3d
    y_train_enc = y_train_enc.reshape((len(y_train_enc), 1, 1))
    y_test_enc = y_test_enc.reshape((len(y_test_enc), 1, 1))
    # prepare each input head
    in_layers = list()
    em_layers = list()
    for i in range(len(X_train_enc)):
    	# calculate the number of unique inputs
    	n_labels = len(np.unique(X_train_enc[i]))
    	# define input layer
    	in_layer = Input(shape=(1,))
    	# define embedding layer
    	# em_layer = Embedding(n_labels, 10)(in_layer)
    	em_layer = Embedding(n_labels, len(set(X_train_enc[i]))//2)(in_layer)
    	# store layers
    	in_layers.append(in_layer)
    	em_layers.append(em_layer)
    # concat all embeddings
    merge = concatenate(em_layers)
    dense = Dense(10, activation='relu', kernel_initializer='he_normal')(merge)
    output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=in_layers, outputs=output)
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot graph
    plot_model(model, show_shapes=True, to_file='embeddings.png')
    # fit the keras model on the dataset
    model.fit(X_train_enc, y_train_enc, epochs=20, batch_size=16, verbose=2)
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))
else:
    # prepare input data
    X_train_enc, X_test_enc = prepare_inputs_oe(X_train, X_test)
    # prepare output data
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    
    # define the model
    model = Sequential()
    model.add(Dense(10, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X_train_enc, y_train_enc, epochs=100, batch_size=16, verbose=2)
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))