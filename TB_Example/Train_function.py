# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 19:18:48 2021

@author: Lin
"""


#Train function

import os
import cv2
import timeit
import time, math
import Function as f
import Image_function as IF
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import Callback 
from tensorflow.keras.models import load_model

def train_model():
    conv_base = ResNet50(weights = 'imagenet',
                         include_top = False,  #whether to include the fully-connected layer at the top of the network.
                         input_shape = (224, 224, 3))

    conv_base.trainable = True

    model = models.Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    
    model.compile(loss='binary_crossentropy', 
              optimizer= optimizers.Adam (lr = 1e-4, name = 'adam'),
                 metrics = ['acc'])
    
    return model

def sample_train_data(train_df, train_ratio = 0.7):

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    x_train, x_val, y_train, y_val = train_test_split(train_df['Filepath'], 
                                                        train_df['Label'], 
                                                        test_size = 1 - train_ratio)

    sub_train_df = pd.DataFrame(columns=['Filepath', 'Label'])
    sub_train_df['Label'] = y_train
    sub_train_df['Filepath'] = x_train

    sub_val_df = pd.DataFrame(columns=['Filepath', 'Label'])
    sub_val_df['Label'] = y_val
    sub_val_df['Filepath'] = x_val
    
    return sub_train_df, sub_val_df

def prediction_df(test_df, model):  #resize 224
    
    result_normal = []
    result_tb = []
    
    print('test dataset predicting...')
    for index in test_df.index:
        
        filepath = test_df.Filepath[index]
        test_img = cv2.imread(filepath)
        
        #original prediction
        test_img = test_img.reshape((1,) + test_img.shape)
        prob = model.predict(test_img)
        result_tb.append(prob[0][0])
        result_normal.append(1-prob[0][0])

    test_df['Normal_Prob'] = result_normal 
    test_df['TB_Prob'] = result_tb 
    
    return test_df