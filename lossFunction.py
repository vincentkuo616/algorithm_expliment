# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:36:44 2020

@author: vincentkuo
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mse(true, pred):
    """
    true: array of true values
    pred: array of predicted values
    
    returns: mean square error loss
    """
    
    return np.sum((true - pred)**2)

def mae(true, pred):
    """
    true: array of true values
    pred: array of predicted values
    
    returns: mean absolute error loss
    """
    
    return np.sum(np.abs(true - pred))

def sm_mae(true, pred, delta):
    """
    true: array of true values
    pred: array of predicted values
    
    returns: smoothed mean absolute error loss
    """
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)

def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)

def quan(true, pred, theta):
    loss = np.where(true >= pred, theta*(np.abs(true-pred)), (1-theta)*(np.abs(true-pred)))
    return np.sum(loss)

def quanl2(true, pred, theta):
    loss = np.where(true >= pred, theta*((true - pred)**2), (1-theta)*((true - pred)**2))
    return np.sum(loss)

def fair(true, pred, constant):
    loss = constant**2*((np.abs(true-pred)/constant)-np.log((np.abs(true-pred)/constant)+1))
    return np.sum(loss)

def mad(true, pred):
    return np.median(np.abs(true - pred))

fig, ax1 = plt.subplots(1,1, figsize = (10,6.5))

target = np.repeat(0, 1000)
pred = np.arange(-10,10, 0.02)

# calculating loss function for all predictions.
loss_mse = [mse(target[i], pred[i]) for i in range(len(pred))]
loss_mae = [mae(target[i], pred[i]) for i in range(len(pred))]
loss_sm_mae1 = [sm_mae(target[i], pred[i], 5) for i in range(len(pred))]
loss_sm_mae2 = [sm_mae(target[i], pred[i], 10) for i in range(len(pred))]
loss_logcosh = [logcosh(target[i], pred[i]) for i in range(len(pred))]
loss_quan1 = [quan(target[i], pred[i], 0.25) for i in range(len(pred))]
loss_fair = [fair(target[i], pred[i], 1) for i in range(len(pred))]
loss_fair5 = [fair(target[i], pred[i], 5) for i in range(len(pred))]
loss_mad = [mad(target[i], pred[i]) for i in range(len(pred))]


losses = [loss_mse, loss_mae, loss_sm_mae1, loss_sm_mae2, loss_logcosh, loss_quan1, loss_fair, loss_fair5, loss_mad]
names = ['MSE', 'MAE','Huber (5)', 'Huber (10)', 'Log-cosh', 'Quantile (0.25)', 'fair(1)', 'fair(5)', 'MAD']
cmap = ['#d53e4f',
'#fc8d59',
'#fee08b',
'#e6f598',
'#99d594',
'#3288bd',
'#333333',
'#1770ff',
'#cccccc']

for lo in range(len(losses)):
    ax1.plot(pred, losses[lo], label = names[lo], color= cmap[lo])
ax1.set_xlabel('Residues')
ax1.set_ylabel('Loss')
ax1.set_title("Loss with Residues values")
ax1.legend()
ax1.set_ylim(bottom=0, top=40)