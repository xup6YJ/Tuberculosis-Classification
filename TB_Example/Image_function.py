# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:40:49 2021

@author: Lin
"""

import cv2
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

#Image Preprocessing Function
def random_crop_equalizeHist(img, resize = 256, train = True):
    
    if train:
        img2 = cv2.resize(img, (resize, resize), interpolation = cv2.INTER_LINEAR)
        row = random.randint(0,32)
        col = random.randint(0,32)
        r1 = row + 0
        r2 = row + 0 + resize - 32
        c1 = col + 0
        c2 = col + 0 + resize - 32
        img2 = img2[r1:r2, c1:c2]
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        img2 = cv2.equalizeHist(img2)
        img2 = cv2.merge((img2, img2, img2))
    else:
        img2 = cv2.resize(img, (resize - 32, resize - 32), interpolation = cv2.INTER_LINEAR)
    
    return img2

def crop_up_down_equalizeHist(img, resize = 256, train = False):
    
    if train:
        img2 = cv2.resize(img, (resize, resize), interpolation = cv2.INTER_LINEAR)
        img2 = img2[8:232, 16:240]
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        img2 = cv2.equalizeHist(img2)
        img2 = cv2.merge((img2, img2, img2))
    else:
        img2 = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        img2 = cv2.equalizeHist(img2)
        img2 = cv2.merge((img2, img2, img2))
    
    return img2

def datagen(rescale = True):
    if rescale:
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 0,
            width_shift_range = 0,
            height_shift_range = 0,
            shear_range = 0,
            zoom_range = 0.01,
            horizontal_flip = True,
            fill_mode = 'nearest')

        test_datagen = ImageDataGenerator(rescale = 1./255)
        print('The pictures had been Rescaled')
        
    else:
        train_datagen = ImageDataGenerator(
            rotation_range = 0,
            width_shift_range = 0,
            height_shift_range = 0,
            shear_range = 0,
            zoom_range = 0.01,
            horizontal_flip = True,
            fill_mode = 'nearest')

        test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
        print('The pictures were Non - Rescaled')

    return train_datagen, test_datagen
