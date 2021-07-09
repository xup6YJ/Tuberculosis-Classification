# -*- coding: utf-8 -*-

import Function as f
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

history_path = os.path.join(os.getcwd(), 'bootstrap_result/history')  #correct path
result_path = os.path.join(os.getcwd(), 'bootstrap_result/pred_result')  #correct path

history_file = f.count_files(history_path)
history_file

#Concat all history files
all_history = pd.DataFrame()
for i in range(len(history_file)):
    history_df = pd.read_csv(os.path.join(history_path, history_file[i]))
    history_df['bootstrap'] = i+1
    history_df = history_df.rename(columns={'Unnamed: 0': 'epoch'})
    history_df.epoch = history_df.epoch + 20*i + 1
    all_history = pd.concat([all_history, history_df], ignore_index = True)
    
#Plot history
#plot val_acc, val_loss
history = all_history
acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) +1)
# epochs= history['bootstrap']

plt.plot(epochs, acc, 'b', label = 'Training acc')
plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.figure()
    
def result_original_matrix(table, Bootstrap, criteria):
    
    tabl2_new = np.zeros((2,2))
    tabl2_new[0, 0] = int(table[1, 1])
    tabl2_new[1, 1] = int(table[0, 0])
    tabl2_new[1, 0] = int(table[1, 0]) #original False Negative
    tabl2_new[0, 1] = int(table[0, 1]) #original False Positive

    bot = ['True Positive', 'True Negative']
    left = ['Model Positive', 'Model Negative']

    df_cm = pd.DataFrame(tabl2_new, 
                         index = [i for i in left],
                         columns = [i for i in bot])
    
    result_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                      'Accuracy', 'Loading Rate','criteria', 'Bootstrap'])
    result_df.TP = pd.Series(int(tabl2_new[0,0]))
    result_df.FP = pd.Series(int(tabl2_new[0,1]))
    result_df.TN = pd.Series(int(tabl2_new[1,1]))
    result_df.FN = pd.Series(int(tabl2_new[1,0]))
    result_df.Sensitivity = tabl2_new[0, 0]/(tabl2_new[0, 0] + tabl2_new[1, 0])
    result_df.Specificity = tabl2_new[1, 1]/(tabl2_new[0, 1] + tabl2_new[1, 1])
    result_df.PPV = tabl2_new[0, 0]/(tabl2_new[0, 0] + tabl2_new[0, 1])
    result_df.NPV = tabl2_new[1, 1]/(tabl2_new[1, 1] + tabl2_new[1, 0])
    result_df.Accuracy = (tabl2_new[0, 0] + tabl2_new[1, 1])/table.sum()
    result_df.criteria = criteria
    result_df.Bootstrap = Bootstrap

    if(tabl2_new[1, 0] == 0):
        result_df['Loading Rate'] = (tabl2_new[0, 0] + tabl2_new[0, 1])/table.sum()
    else:
        result_df['Loading Rate'] = 'NA'
        
    return result_df

#Result Main
result_file = f.count_files(result_path)

all_tb_result = pd.DataFrame()

for k in range(len(result_file)):
    print('No. ', k)
    result_df = pd.read_csv(os.path.join(result_path, result_file[k]))

    label_tb = []

    for i in range(len(result_df)):

        if result_df.Label[i] == 'TB':
            label_tb.append(1)    
        else:
            label_tb.append(0)

    result_df['Label_TB'] = label_tb 

    #ROC plot
    predict_df_all = result_df

    #tb
    fpr, tpr, thresholds = roc_curve(predict_df_all['Label_TB'], predict_df_all['TB_Prob'])

    fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 80)

    ax1.plot(fpr, tpr, 'b.-', label = 'TB (AUC:%2.2f)' % roc_auc_score(predict_df_all['Label_TB'], predict_df_all['TB_Prob']))
    ax1.legend(loc = 4)
    ax1.set_xlabel('1 - Specificity')
    ax1.set_ylabel('Sensitivity')
    plt.title('Bootstrap' + str(k+1))

    rocplot_path = os.path.join(os.getcwd(), 'bootstrap_result/rocplot')  #correct path
    if not os.path.isdir(rocplot_path): os.mkdir(rocplot_path)
    fig.savefig(os.path.join(rocplot_path,'Bootstrap' + str(k+1) + '.JPEG'))

    #Confusion matrix, ROC threshold
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    index = roc.iloc[(roc.tf-0).abs().argsort()[:1]].index
    thresh_tb = float(thresholds[index])

    result_tb = []

    for i in range(len(predict_df_all)):
        if predict_df_all.TB_Prob[i] >= thresh_tb:
            result_tb.append(1)
        else:
            result_tb.append(0)
            
    predict_df_all['Result_TB'] = result_tb
    
    #Table
    table_tb = confusion_matrix(predict_df_all['Label_TB'], predict_df_all['Result_TB'])
    tb_df = result_original_matrix(table_tb, Bootstrap = k, criteria = thresh_tb)
        
    
tb_df_path = os.path.join(os.getcwd(), 'Bootstrap_pred_result.csv')
tb_df.to_csv(tb_df_path, index = False)
