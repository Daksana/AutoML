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
import category_encoders as ce
import seaborn as sns
import dataframe_image as dfi
#from main2 import *
#Display columns which have outliers with plot ex: Histogram,Boxplot

def colvisualization_before(data,column):
    #boxplot
    fig = plt.figure(figsize=(10,5))
    sns.boxplot(data[column])
    plt.title("Box Plot:"+column, fontsize=15)
    plt.xlabel(column, fontsize=14)
    direc="/Users/xfach/Senzmate/upload_filecode/plots/"
    plt.savefig(direc+column+" boxplot.png")
        
    #"Histogram":
    data[column].hist()
    plt.savefig(direc+column+" histogram.png")
#Display columns after removal of outliers with plot ex: Histogram,Boxplot
def colvisualization_after(data,column):
    #boxplot
    fig = plt.figure(figsize=(10,5))
    sns.boxplot(data[column])
    plt.title("Box Plot:"+column, fontsize=15)
    plt.xlabel(column, fontsize=14)
    direc="/Users/xfach/Senzmate/upload_filecode/plots/"
    plt.savefig(direc+column+" outlierremoved_boxplot.png")
        
    #"Histogram":
    data[column].hist()
    plt.savefig(direc+column+" outlierremoved_histogram.png")    
    
#Display columns which have outliers with measurements ex: Skewness/Interquartile Range/Standard Deviation
def outlierstatistics(data,column):
    #skewness value
   # print('skewness value',data[column].skew())   #within the range of -1 to 1
    
    #IQR
    direc="/Users/xfach/Senzmate/upload_filecode/plots/"
    Q1=data[column].quantile(0.25)
    Q3=data[column].quantile(0.75)
    IQR=Q3-Q1
    whisker_width = 1.5
    col_outliers_iqr = data[(data[column] < Q1 - whisker_width*IQR) | (data[column] > Q3 + whisker_width*IQR)]
    dfi.export(col_outliers_iqr,direc+column+" outlier_df_iqr.png")
    #saving dataframe in above mentioned location
    
    #stddeviation
    col_mean = data[column].mean()
    col_std = data[column].std()
    low= col_mean -(3 * col_std)
    high= col_mean + (3 * col_std)
    col_outliers_sd = data[(data[column] < low) | (data[column] > high)]
    dfi.export(col_outliers_sd,direc+column+" outlier_df_sd.png")
  


#removing outliers
def removeoutliers(data,column, method):
    r_column=column
    if method == "Flooring And Capping":
        Q1 = data[r_column].quantile(0.25)
        Q3 = data[r_column].quantile(0.75)
        IQR = Q3 - Q1
        whisker_width = 1.5
        lower_whisker = Q1 -(whisker_width*IQR)
        upper_whisker = Q3 +(whisker_width*IQR)
        data[r_column]=np.where(data[r_column]>upper_whisker,upper_whisker,np.where(data[r_column]<lower_whisker,lower_whisker,data[r_column]))
        
    elif method == "Trimming":
        Q1 = data[r_column].quantile(0.10)
        Q3 = data[r_column].quantile(0.90)
        IQR = Q3 - Q1
        whisker_width = 1.5
        lower_whisker = Q1 - (whisker_width*IQR)
        upper_whisker = Q3 + (whisker_width*IQR)
        index=data[r_column][(data[r_column]>upper_whisker)|(data[r_column]<lower_whisker)].index
        data.drop(index,inplace=True)

def main(or_data,or_columnname,or_removalmethod):
    data=or_data
    column=or_columnname
    removal_method=or_removalmethod
    #create dictionary to store final results
    colvisualization_before(data,column)
    outlierstatistics(data,column)
    removeoutliers(data,column,removal_method)
    colvisualization_after(data,column)
    #return Dataframe after outlier removal
    return data
# if __name__=="__main__":
#     main(
#         or_data   ,
#         or_columnname    ,
#         or_removalmethod    
#         )
        
    
    



