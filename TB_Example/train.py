# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:14:49 2021

@author: Lin
"""
import os
import cv2
import timeit
import time, math
import Function as f
import Image_function as IF
import pandas as pd
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras import models, layers, optimizers


import Train_function as TF

#Train Dataframe
train_path_normal = 'D:/File_X/Graduate/TB/Modeling/train/train_normal'  #correct path
train_path_tb = 'D:/File_X/Graduate/TB/Modeling/train/train_tb'  #correct path

nor_file = f.count_files(train_path_normal)
tb_file = f.count_files(train_path_tb)
df_normal = f.file2dataframe(files = nor_file, Label = 'Normal', path = train_path_normal)
df_tb = f.file2dataframe(files = tb_file, Label = 'TB', path = train_path_tb)

train_df = df_normal.append(df_tb)
train_df

#Test Dataframe
predict_path_normal = 'D:/File_X/Graduate/TB/Modeling/test/test_normal'  #correct path
predict_path_tb = 'D:/File_X/Graduate/TB/Modeling/test/test_tb'  #correct path

nor_file = f.count_files(predict_path_normal)
tb_file = f.count_files(predict_path_tb)
df_normal = f.file2dataframe(files = nor_file, Label = 'Normal', path = predict_path_normal)
df_tb = f.file2dataframe(files = tb_file, Label = 'TB', path = predict_path_tb)

test_df = df_normal.append(df_tb)
test_df

#Data Generator
train_datagen, test_datagen = IF.datagen(rescale = False)

#Model Architecture
model = TF.train_model()
model.summary()

def find_best_model(model_path):
    
    last_epoch_model = f.count_files(model_path)
    model_accuracy = []
    for i in range(len(last_epoch_model)):
        model_accuracy.append(int(last_epoch_model[i][9:11]))
        index = model_accuracy.index(max(model_accuracy))

    return last_epoch_model[index]

#bootstrapping main
def bootstrap_main(bootstrap, epochs):
    for i in range(bootstrap):

        print('No.', i, ' bootstrap')
        
        #Result Saving Directory
        result_dir = os.path.join(os.getcwd(), 'bootstrap_result')  #correct path
        if not os.path.isdir(result_dir): os.mkdir(result_dir)
        result_model_dir = os.path.join(result_dir, 'model')   #correct path
        if not os.path.isdir(result_model_dir): os.mkdir(result_model_dir)
        
        if i != 0:

            path = os.path.join(result_model_dir, 'bootstrap' + str(i-1)) #correct path           
            model_path = find_best_model(path)
            model_path = os.path.join(path, model_path)

            try:
                model = load_model(model_path)
                print("Model Load Success!", model_path)
            except:
                print('Model Load Failure')

            model.compile(loss='binary_crossentropy', 
                          optimizer= optimizers.Adam (lr = 1e-4, name = 'adam'),
                          metrics = ['acc'])

        else:

            model = TF.train_model()

        #subtrain, subvalidation dataframe
        sub_train_df, sub_val_df = TF.sample_train_data(train_df = train_df)

        #Data generator
        train_batch_size = 8
        validation_batch_size = 4

        train_df_gen = train_datagen.flow_from_dataframe(dataframe = sub_train_df,
                                      x_col = 'Filepath',
                                      y_col = 'Label',
                                      weight_col = None,
                                      target_size = (224, 224),
                                      class_mode = "binary",
                                      shuffle =  True,
                                      batch_size = train_batch_size,
                                      interpolation = "bilinear")

        val_df_gen = test_datagen.flow_from_dataframe(
                                    dataframe = sub_val_df,
                                    x_col = 'Filepath',
                                    y_col = 'Label',
                                    target_size = (224, 224),
                                    interpolation="bilinear",
                                    batch_size = validation_batch_size,
                                    class_mode = 'binary',
                                    shuffle = False)
        
        # Check Label
        label_map = (train_df_gen.class_indices)
        print(label_map)

        # Callbalck Function
        model_dir = os.path.join(result_model_dir, 'bootstrap' + str(i)) #correct path  
        if not os.path.isdir(model_dir): os.mkdir(model_dir)

        model_path = os.path.join(model_dir, 'ResNet50-{epoch:02d}-{val_acc:.4f}.h5') #correct path

        checkpoint = ModelCheckpoint(model_path,
                                     monitor = 'val_loss',
                                     verbose = 1,
                                     save_best_only = True,
                                     mode = 'min')

        #learning rate scheduler
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                      factor = 0.1,
                                      patience = 3,
                                      min_lr = 1e-7,
                                      verbose = 1,
                                      model = 'min',
                                      min_delta = 0.001)
        
        #Early Stopping
        early = EarlyStopping(monitor = "val_loss", 
                              mode = "min", 
                              patience = 9)

        callbacks_list = [checkpoint, reduce_lr, early]

        train_n = len(sub_train_df)
        validation_n = len(sub_val_df)
        
        #Start training
        start = timeit.default_timer()

        history = model.fit_generator(train_df_gen,
                                      steps_per_epoch = train_n // train_batch_size,
                                      epochs = epochs,
                                      validation_data = val_df_gen,
                                      validation_steps = validation_n // validation_batch_size,
                                      callbacks = callbacks_list)
        
        stop = timeit.default_timer()
        time = stop - start
        
        #Save history
        hist_df = pd.DataFrame(history.history)
        hist_df['Training Time'] = time
        
        hist_dir = os.path.join(result_dir, 'history')  #correct path
        if not os.path.isdir(hist_dir): os.mkdir(hist_dir)
        
        if i < 10:
            his_path = os.path.join(hist_dir, 'bootstrap'+ '0' + str(i) + '.csv') #correct path
        else:
            his_path = os.path.join(hist_dir, 'bootstrap' + str(i) + '.csv')  #correct path
        with open(his_path, mode = 'w') as f:
            hist_df.to_csv(f)
        
        
        #Prediction  
        model_path = find_best_model(model_dir)
        model_path = os.path.join(model_dir, model_path)

        try:
            model = load_model(model_path)
            print("Model Load Success!", model_path)
        except:
            print('Model Load Failure')
            
        predict_df = TF.prediction_df(test_df = test_df, model = model)
        
        pred_result_dir = os.path.join(result_dir, 'pred_result')   #correct path
        if not os.path.isdir(pred_result_dir): os.mkdir(pred_result_dir)
        
        if i < 10:
            predict_df_path = os.path.join(pred_result_dir, 'result_epoch' + '0' + str(i) + '.csv') #correct path
        else:
            predict_df_path = os.path.join(pred_result_dir, 'result_epoch' + str(i) + '.csv') #correct path
            
        predict_df.to_csv(predict_df_path, index = False)

bootstrap_main(bootstrap = 1, epochs = 15)  #bootstrap = 1, epochs = 15 means no bootstrap model training for 15 epochs)
