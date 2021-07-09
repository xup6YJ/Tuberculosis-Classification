# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:09:25 2021

@author: Lin
"""

import os
import pandas as pd  
from pandas import DataFrame, Series

#Function

#Count files in folder
def count_files(path):
    onlyfiles = next(os.walk(path))[2] #dir is your directory path as string
    return onlyfiles


def file2dataframe(files, Label, path):
    df = pd.DataFrame(columns=['Filename', 'Filepath', 'Label'], index = files)
    for i in range(len(files)):
        df['Filename'][i] = files[i]
        df['Filepath'][i] = os.path.join(path, files[i])
        df['Label'][i] = Label
        
    return df



