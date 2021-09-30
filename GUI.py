# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:06:07 2021

@author: Administrator
"""

from tkinter import *
import tkinter as tk
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
# loading Python Imaging Library 
from PIL import ImageTk, Image   
# To get the dialog box to open when required  
from tkinter import filedialog
import cv2
from tkinter import messagebox as msg
import os
import pydicom as dicom
import matplotlib.pyplot as plt

import pandas as pd
from pydicom import dcmread
from pydicom.data import get_testdata_file
import numpy as np
import png, os, pydicom
import cv2
import os
import pydicom
import glob
import gdcm
import shutil
from tensorflow.keras.models import load_model

model = load_model('0403ResNet101-08-0.99186438.h5')
print('Model Loaded Sucessfully')

import pandas as pd

def dicom2png(inputfile):
    try:
        ds = pydicom.dcmread(inputfile)
        shape = ds.pixel_array.shape
        img = ds.pixel_array # get image array
        outputfile = inputfile + '.png'
        cv2.imwrite(outputfile, img) # write png image
#         print('Convert Succeed: ', outputfile)
    except:
        print('Could not convert: ', inputfile)
        
def open_img(): 
    # Select the Imagename  from a folder  
    x = openfilename() 
    # opens the image 
    
def open_img(): 
    # Select the Imagename  from a folder  
    x = openfilename() 
    # opens the image 
    img = Image.open(x) 
    im1 = img.save("casting.jpeg")
    img = ImageTk.PhotoImage(img) 
    # create a label 
    panel = Label(root, image = img)   
    # set the image as img  
    panel.image = img 
    panel.place(bordermode=OUTSIDE, x=50, y=50)
    
def openfilename(): 
    # open file dialog box to select image 
    # The dialogue box has a title "Open" 
    filename = filedialog.askopenfilenames(title ='Select Image') 
    return filename

def askpath():
    path = filedialog.askdirectory() 
    return path

def prediction():
    
    result_normal = []
    result_pneumonia = []
    result_tb = []
    
    filenames = openfilename() 
    df_result = pd.DataFrame(columns=['Normal_Prob', 'Suspected_Prob', 'TB_Prob', 'Result'])
    df_result["Filename"] = filenames
    
    for image in filenames:
        
        if image[-3:] != 'png':
            dicom2png(image)
            test_image = cv2.imread(image + '.png')
        else:
            test_image = cv2.imread(image)
            
        test_image = cv2.resize(test_image, (224, 224), interpolation = cv2.INTER_LINEAR)
        test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
        test_image = cv2.equalizeHist(test_image)
        test_image = cv2.merge((test_image, test_image, test_image))
        test_img = test_image.reshape((1,) + test_image.shape)
        
        ###predict
        prob = model.predict(test_img)
        result_normal.append(prob[0][0])
        result_pneumonia.append(prob[0][1])
        result_tb.append(prob[0][2])
        
        if image[-3:] != 'png':
            os.remove(image + '.png')
        
    df_result['Normal_Prob'] = result_normal 
    df_result['Suspected_Prob'] = result_pneumonia 
    df_result['TB_Prob'] = result_tb
    
    #Labeling
    prob_pneumonia = []
    prob_tb = []

    for i in range(len(df_result)):
        if df_result.Suspected_Prob[i] >= 0.927692235:
            prob_pneumonia.append(1)
        else:
            prob_pneumonia.append(0)

        if df_result.TB_Prob[i] >= 0.05518347:
            prob_tb.append(1)
        else:
            prob_tb.append(0)
    
    df_result['Result_Suspected'] = prob_pneumonia
    df_result['Result_TB'] = prob_tb
    
    prob_result = []
    for i in range(len(df_result)):
        if df_result.Result_Suspected[i] + df_result.Result_TB[i] > 0:
            prob_result.append('Abnormal')
        else:
            prob_result.append('Normal')

    df_result['Result'] = prob_result
    
    num_ab = elm_count = prob_result.count('Abnormal')
    result = 'There are ', num_ab, 'abnormal files'
    result = Label(root, text = result) 
    
    # set the image as img  
    result.place(bordermode = OUTSIDE, x = 400, y = 120)
    msg.showinfo('Tip', 'Finished predicting, please download the result csv file')
    
    #make paths
    path = askpath()
    target_dir = path + '/target file'
    tb_dir = target_dir + '/TB'
    suspected_dir = target_dir + '/Suspected_TB'
    if not os.path.isdir(target_dir): os.mkdir(target_dir)
    if not os.path.isdir(tb_dir): os.mkdir(tb_dir)
    if not os.path.isdir(suspected_dir): os.mkdir(suspected_dir)
    
    #download CSV
    df_result.to_csv(os.path.join(target_dir, 'TB_result.csv'), index = True)
    msg.showinfo('Tip', 'Finished downloading')
    
    #TB files convert
    ab_file = df_result[df_result['Result'] == 'Abnormal']

    for i in range(len(ab_file)):
        
        fname = ab_file['Filename'][i]
        
        if ab_file['Result_TB'][i] == 1:
            TB_Prob = ab_file['TB_Prob'][i]
            prob = round(TB_Prob, 3)
            src = fname
            pos = src.rfind('/')
            dst = tb_dir + '/' + str(prob) + src[pos+1:] #convert to tb_dir
            print(src, dst)
            shutil.copyfile(src, dst)
        else:
            Suspected_Prob = ab_file['Suspected_Prob'][i]
            prob = round(Suspected_Prob, 3)
            src = fname
            pos = src.rfind('/')
            dst = suspected_dir + '/' + str(prob) + src[pos+1:] #convert to suspected_dir
            print(src, dst)
            shutil.copyfile(src, dst)
            
def count_files(path):
    onlyfiles = next(os.walk(path))[2] #dir is your directory path as string
    return onlyfiles

# APP main 
# Create a window 
root = Tk()   
# Set Title as Image Loader 
root.title("TaoNet")   
# Set the resolution of window 
root.geometry("700x400")   
# Do't Allow Window to be resizable 
root.resizable(width = False, height = False)  

# Create a button and place it into the window using place layout
#########################################################################
#Introduction
introduction = Label(root, text = "TB Chest X-ray Screening")
introduction.place(bordermode=OUTSIDE, x=20, y=20)

#TB image
load = Image.open('TB.png')
render = ImageTk.PhotoImage(load)
img = Label(root, image=render)
img.image = render
img.place(x=50, y=50)

#NDMC image
ndmc_img = Image.open('ndmc2.jpg')
ndmc_render = ImageTk.PhotoImage(ndmc_img)
img2 = Label(root, image=ndmc_render)
img2.image = ndmc_render
img2.place(x=475, y=0)

#Tao image
tao_img = Image.open('tao2.jpg')
tao_render = ImageTk.PhotoImage(tao_img)
img3 = Label(root, image=tao_render)
img3.image = tao_render
img3.place(x=700-150, y=400-150)

#Result
btn_predict = Button(root, text ='Start Predict', command = prediction).place(x = 50, y= 300) 
result_hd = Label(root, text = "RESULT")
result_hd.place(bordermode=OUTSIDE, x=400, y=90)
########################################################################

root.mainloop()

