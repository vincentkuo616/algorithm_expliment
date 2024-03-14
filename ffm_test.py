# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:41:03 2021

@author: vincentkuo
"""


data_path = 'C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\small_train.txt'

import xlearn as xl

# Training task
ffm_model = xl.create_ffm()         # Use field-aware factorization machine (ffm)
ffm_model.setTrain(data_path)

# parameter:
#  0. task: binary classification
#  1. learning rate : 0.2
#  2. regular lambda : 0.002
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
#  param = {'task':'reg', 'lr':0.2, 'lambda':0.002}

# Train model
ffm_model.fit(param, "C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\model.out")

ffm_model.setSigmoid()
# ffm_model.setSign() 將結果轉化為0, 1
ffm_model.setTest("C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\small_test.txt")
ffm_model.predict("C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\model.out","C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\output.txt")