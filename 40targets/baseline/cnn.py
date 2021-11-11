#! user/bin/env python3
# -*- coding: utf-8 -*-

'''
    Project:
        SSVEP-BCI Evaluate

    author: Bo Dai

    version:
        v1.0, 2021, 11.09

'''
from keras import models
from keras.backend import backend
import numpy as np
import scipy.io as sio

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from keras import initializers, regularizers

from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.losses import categorical_crossentropy


# 定义模型参数
CNN_PARAMS = {
    'batch_size': 64,    # 训练的最小batch
    'epochs': 50,    # total number of training epochs/iterations
    'droprate': 0.25,   # dropout ratio
    'learning_rate': 0.001,    # model learning rate
    'lr_delay': 0.0,    # learning rate decay ratio
    'l2_lambda': 0.0001,    # l2 regularrization parameter
    'momentum': 0.9,    # momentum term for stochastic gradient descent optimization.
    'kernel_f': 10,    # 1D kernel to operate on conv_1 layer for the SSVEP CNN
    'n_ch': 8,    # eeg chaneels
    'num_classes': 40    # targets/classes

}

# 确定模型架构
def CNN_model(input_shape, CNN_PARAMS):


    model = Sequential()
    # The first part
    model.add(Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(CNN_PARAMS['n_ch'], 1),
            input_shape=(input_shape[0], input_shape[1], input_shape[2]),
            padding="valid", kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']),
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))
    # The second part
    model.add(Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1, CNN_PARAMS['kernel_f']), 
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="valid", 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))  
    # The last
    model.add(Flatten())
    model.add(Dense(CNN_PARAMS['num_classes'], activation='softmax', 
                    kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    
    return model























