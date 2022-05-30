# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:21:06 2022

@author: Daksana
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
#Visualization
import matplotlib.pyplot as plt
#import category_encoders as ce
import seaborn as sns
import dataframe_image as dfi

def main2(dataframe,outlierremovaldatainputs):
    data_f=dataframe
    #col_list=outlierremovaldatainputs.get("column_list")
    print("data_f:")
    print(data_f)
    t_column_list=outlierremovaldatainputs.get("trimming_column_list")
    
    print(outlierremovaldatainputs.get("trimming_column_list"))
    print("t_column_list:")
    print(t_column_list)
    f_column_list=outlierremovaldatainputs.get("fandc_column_list")
    print("f_column_list:")
    print(f_column_list)
    
    print("data.columns:")
    print(data_f.columns)
    
    if t_column_list:
        for col in t_column_list:
            Q1 = data_f[col].quantile(0.10)
            Q3 = data_f[col].quantile(0.90)
            IQR = Q3 - Q1
            whisker_width = 1.5
            lower_whisker = Q1 - (whisker_width*IQR)
            upper_whisker = Q3 + (whisker_width*IQR)
            index=data_f[col][(data_f[col]>upper_whisker)|(data_f[col]<lower_whisker)].index
            data_f.drop(index,inplace=True)
    
    elif f_column_list:
        for col in f_column_list: 
            Q1 = data_f[col].quantile(0.25)
            Q3 = data_f[col].quantile(0.75)
            IQR = Q3 - Q1
            whisker_width = 1.5
            lower_whisker = Q1 -(whisker_width*IQR)
            upper_whisker = Q3 +(whisker_width*IQR)
            data_f[col]=np.where(data_f[col]>upper_whisker,upper_whisker,np.where(data_f[col]<lower_whisker,lower_whisker,data_f[col]))
    
    print("data.columns:")
    print(data_f.columns)
    
    return data_f;

    
    



