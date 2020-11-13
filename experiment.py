# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:01:37 2018

@author: Chandrasen.Wadikar
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model= Sequential()
model.add(Dense(units=64,activation='relu',input_dim=1424))
model.add(Dense(units=2696))
model.compile(loss='mse',optimizer='adam')

model.fit(predictors[0:80,], estimator[0:80,]),
      validation_data=(predictors[81:,],estimator[81:,]),
      epochs=80,batch_size=32)

np.savetxt("keras_fit.csv",model.predict(data),delimiter=",")