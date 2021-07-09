# -*- coding: utf-8 -*-

import Function as f
import os
import numpy as np  
import pandas as pd  
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

path_normal = os.path.join(os.getcwd(), 'MergeNormal') #path to normal images
path_TB = os.path.join(os.getcwd(), 'MergeTB')  #path to TB images  

nor_file = f.count_files(path_normal)
tb_file = f.count_files(path_TB)

# os.remove(path_abnormal + '/Thumbs.db')  #Remove Thumbs.db if it occured

#Build DataFrame
df_normal = f.file2dataframe(files = nor_file, Label = 'Normal', path = path_normal)
df_tb = f.file2dataframe(files = tb_file, Label = 'TB', path = path_TB)

df_all = df_normal.append(df_tb)

csv_path = 'all_data_id.csv'
df_all.to_csv(csv_path, index = False)

df_all

#Train/ Test Split
train_ratio = 0.7

# train is now 70% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train_normal, x_test_normal, y_train_normal, y_test_normal = train_test_split(df_normal['Filepath'], 
                                                                                df_normal['Label'], 
                                                                                test_size = 1 - train_ratio,
                                                                               random_state = 1)

x_train_tb, x_test_tb, y_train_tb, y_test_tb = train_test_split(df_tb['Filepath'], 
                                                                df_tb['Label'], 
                                                                test_size = 1 - train_ratio,
                                                               random_state = 1)

print('Train Normal: ', len(x_train_normal),
      'Train TB: ', len(x_train_tb),
      'Test Normal: ', len(x_test_normal),
      'Test TB: ', len(x_test_tb),)

#Merge Train Data
x_train_normal_df = pd.DataFrame(x_train_normal)
x_train_normal_df['Label'] = 'Normal'
x_train_normal_df

x_train_tb_df = pd.DataFrame(x_train_tb)
x_train_tb_df['Label'] = 'TB'
x_train_tb_df

train_df = x_train_normal_df.append(x_train_tb_df)
train_df

#Merge Test Data
x_test_normal_df = pd.DataFrame(x_test_normal)
x_test_normal_df['Label'] = 'Normal'
x_test_normal_df

x_test_tb_df = pd.DataFrame(x_test_tb)
x_test_tb_df['Label'] = 'TB'
x_test_tb_df

test_df = x_test_normal_df.append(x_test_tb_df)
test_df

#Save Dataframe
train_csv_path = 'TrainID.csv'
train_df.to_csv(train_csv_path, index = False)

test_csv_path = 'TestID.csv'
test_df.to_csv(test_csv_path, index = False)
