# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 12:56:53 2017

@author: DELL
"""
import xlrd
import numpy as np
import pandas as pd
fle_loc="D:/X_Full.xlsx"
wkb=xlrd.open_workbook('D:/X_Full.xlsx')
sheet=wkb.sheet_by_index(0)
_matrix=[]
for row in range (sheet.nrows):
    _row = []
    for col in range (sheet.ncols):
        _row.append(sheet.cell_value(row,col))
    _matrix.append(_row)
x=np.matrix(_matrix)
#print(x.shape)

fle_loc="D:/Y_train.xlsx"
wkb=xlrd.open_workbook('D:/Y_Full.xlsx')
sheet=wkb.sheet_by_index(0)
_matrix=[]
for row in range (sheet.nrows):
    _row = []
    for col in range (sheet.ncols):
        _row.append(sheet.cell_value(row,col))
    _matrix.append(_row)
y=np.matrix(_matrix)

 


print(x)
#print(y.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,BatchNormalization
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping,LearningRateScheduler,ReduceLROnPlateau,TensorBoard
model1 = Sequential()
model1.add(BatchNormalization(input_shape=x.shape[1:]))
model1.add(Activation('relu'))
print(model1.output_shape)
model1.add(Dense(500,init='glorot_uniform',W_regularizer=None, b_regularizer=None))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dense(300,init='glorot_uniform',W_regularizer=None, b_regularizer=None))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dense(200,init='glorot_uniform',W_regularizer=None, b_regularizer=None))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
#model1.add(Dense(100,init='glorot_uniform',W_regularizer=None, b_regularizer=None))
#model1.add(BatchNormalization())
#model1.add(Activation('relu'))
model1.add(Dense(12,init='glorot_uniform',W_regularizer=None, b_regularizer=None))
model1.add(Activation('relu'))
print(model1.output)
reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.9,
                  patience=3,mode='auto')
model1.compile(loss='mse',
              optimizer='adam',
              metrics=['acc'])
model1.fit(x, y,batch_size=32,
                        nb_epoch=300, callbacks=[reduce_lr])
model1.save('d:/model12.h5') 
model1.summary()
print(model1.get_weights())