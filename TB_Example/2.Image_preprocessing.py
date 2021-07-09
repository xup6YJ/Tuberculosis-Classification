# -*- coding: utf-8 -*-

#Image Preprocessing

#read csv
import pandas as pd
from matplotlib import pyplot as plt
import shutil
import os
import Image_function as IF
import cv2

train_csv_path = 'TrainID.csv'
test_csv_path = 'TestID.csv'

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

#Plot Example
i = 1
img_path = os.path.join(os.getcwd(), 'MergeTB/TB (1).png')
im = cv2.imread(img_path)
img3 = IF.crop_up_down_equalizeHist(im, train = True)

plt.imshow(im, cmap='gray', interpolation = 'bilinear')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.imshow(img3, cmap='gray', interpolation = 'bilinear')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

#Image Preprocess

#Transfer file path
original_dir = os.path.join(os.getcwd(), 'Modeling')
if not os.path.isdir(original_dir): os.mkdir(original_dir)
train_dir = os.path.join(original_dir, 'train')
if not os.path.isdir(train_dir): os.mkdir(train_dir)
test_dir = os.path.join(original_dir, 'test')
if not os.path.isdir(test_dir): os.mkdir(test_dir)

train_normal_dir = os.path.join(train_dir, 'train_normal')
if not os.path.isdir(train_normal_dir): os.mkdir(train_normal_dir)   
train_tb_dir = os.path.join(train_dir, 'train_tb')
if not os.path.isdir(train_tb_dir): os.mkdir(train_tb_dir)
     
test_normal_dir = os.path.join(test_dir, 'test_normal')
if not os.path.isdir(test_normal_dir): os.mkdir(test_normal_dir)
test_tb_dir = os.path.join(test_dir, 'test_tb')
if not os.path.isdir(test_tb_dir): os.mkdir(test_tb_dir)

x_train = train_df.Filepath
x_train

#Convert eqal_files
fail_list = []

#Train
for i in range(len(train_df)):
    
    #27 may need to modify
    #find the letter 'N' in your path, the number 27 means 'N' is at the 27th letter of path
    if x_train[i][27]  == 'N':  #Normal
            
        img = cv2.imread(x_train[i])
        img2 = IF.crop_up_down_equalizeHist(img, train = True)
        new_filename = os.path.join(train_normal_dir, x_train[i][34:])
        status = cv2.imwrite(new_filename, img2)
        if status == False:
            fail_list.append(x_train[i])
                
    else:  #TB
        
        img = cv2.imread(x_train[i])
        
        #Augmentation 
        img2 = IF.crop_up_down_equalizeHist(img, train = True)
        img3 = IF.crop_up_down_equalizeHist(img, train = False)
        img4 = IF.random_crop_equalizeHist(img, resize = 256, train = True)
        
        #30 may need to modify, example filename will be 'filename.png'
        new_filename = os.path.join(train_tb_dir, x_train[i][30:])  
        new_filename2 = os.path.join(train_tb_dir, x_train[i][30:-4]+ '_2.png')
        new_filename3 = os.path.join(train_tb_dir, x_train[i][30:-4]+ '_3.png')
        
        #write image
        status = cv2.imwrite(new_filename, img2)
        status2 = cv2.imwrite(new_filename2, img3)
        status3 = cv2.imwrite(new_filename3, img4)
        if status == False | status2 == False | status3 == False :
            fail_list.append(x_train[i])
                    
    if len(fail_list)>0:
        print(fail_list)
        break
    
#Test
fail_list = []
x_test = test_df.Filepath

for i in range(len(test_df)):
    
    if x_test[i][27]  == 'N':  #Normal, TB
            
        img = cv2.imread(x_test[i])
        img2 = IF.crop_up_down_equalizeHist(img)
        new_filename = os.path.join(test_normal_dir, x_test[i][34:]) #34 may need to modify
        status = cv2.imwrite(new_filename, img2)
        if status == False:
            fail_list.append(x_test[i])
                
    else:
        img = cv2.imread(x_test[i]) 
        img2 = IF.crop_up_down_equalizeHist(img)
        new_filename = os.path.join(test_tb_dir, x_test[i][30:]) #30 may need to modify
        status = cv2.imwrite(new_filename, img2)

        if status == False :
            fail_list.append(x_test[i])
                    
    if len(fail_list)>0:
        print(fail_list)
        break
