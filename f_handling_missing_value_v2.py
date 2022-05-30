#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
#import category_encoders as ce
#Visualization
import matplotlib.pyplot as plt

def main1(data,handlingnulldatainputs):
    dataf = data
    
    
    numerical_columnlist=handlingnulldatainputs.get("numerical_column_list")
    categorical_columnlist=handlingnulldatainputs.get("categorical_column_list")
    target_column=handlingnulldatainputs.get("target_column_list")
  #  N_Removecolumn_list=handlingnulldatainputs.get("filldata_numerical").get("remove_column_list")
    
    N_Removecolumn_list=handlingnulldatainputs.get("filldata_numerical").get("remove_column_list")
    
    N_Removerows_list=handlingnulldatainputs.get("filldata_numerical").get("remove_rows_list")
    N_Replacewithzero=handlingnulldatainputs.get("filldata_numerical").get("replace_with_zero")
    N_Replacewithmean=handlingnulldatainputs.get("filldata_numerical").get("replace_with_mean")
    N_Replacewithmedian=handlingnulldatainputs.get("filldata_numerical").get("replace_with_median")
    N_Replacewithmode=handlingnulldatainputs.get("filldata_numerical").get("replace_with_mode")
    N_Replacewithna=handlingnulldatainputs.get("filldata_numerical").get("replace_with_fillna")
    
    C_Removecolumn_list=handlingnulldatainputs.get("filldata_categorical").get("remove_column_list")
    C_Removerows_list=handlingnulldatainputs.get("filldata_categorical").get("remove_rows_list")
    C_Replacewithmode=handlingnulldatainputs.get("filldata_categorical").get("replace_with_mode")
    C_Replacewithna=handlingnulldatainputs.get("filldata_categorical").get("replace_with_fillna")
    
    T_Removerows_list=handlingnulldatainputs.get("filldata_target").get("removerows_list")
    T_Replacewithmode=handlingnulldatainputs.get("filldata_target").get("replace_with_mode")
    T_Replacewithna=handlingnulldatainputs.get("filldata_target").get("replace_with_fillna")
    
    #"filldata_numerical").get("Replacewithzero")
    if handlingnulldatainputs.get("filldata_numerical").get("replace_with_zero")!="":
       columnslist=handlingnulldatainputs.get("filldata_numerical").get("replace_with_zero")
       for col in columnslist:
           dataf[col]=dataf[col].fillna(0)

    #"filldata_numerical").get("Replacewithmean")
    if handlingnulldatainputs.get("filldata_numerical").get("replace_with_mean")!="":
        columnslist=handlingnulldatainputs.get("filldata_numerical").get("replace_with_mean")
       
        for col in columnslist:
            dataf[col]=dataf[col].fillna(dataf[col].mean())
    
    #"filldata_numerical").get("Replacewithmode")
    if handlingnulldatainputs.get("filldata_numerical").get("replace_with_mode")!="":
        columnslist=handlingnulldatainputs.get("filldata_numerical").get("replace_with_mode")
       
        for col in columnslist:
            dataf[col]=dataf[col].fillna(dataf[col].mode()[0])
    
    #"filldata_numerical").get("Replacewithmedian")
    if handlingnulldatainputs.get("filldata_numerical").get("replace_with_median")!="":
        columnslist=handlingnulldatainputs.get("filldata_numerical").get("replace_with_median")
       
        for col in columnslist:
            dataf[col]=dataf[col].fillna(dataf[col].median())
    
    
    #"filldata_numerical").get("Replacewithna")
  
    if handlingnulldatainputs.get("filldata_numerical").get("replace_with_fillna")!="":
        columnslist=handlingnulldatainputs.get("filldata_numerical").get("replace_with_fillna")
        
        for col in columnslist:
            dataf[col]=dataf[col].fillna(np.nan)
    #"filldata_numerical").get("Replacewithna")
  
    if handlingnulldatainputs.get("filldata_categorical").get("replace_with_fillna")!="":
        columnslist=handlingnulldatainputs.get("filldata_categorical").get("replace_with_fillna")
       
        for col in columnslist:
            dataf[col]=dataf[col].fillna(np.nan)
    
    #"filldata_numerical").get("Replacewithna")
  
    if handlingnulldatainputs.get("filldata_target").get("replace_with_fillna")!="":
        columnslist=handlingnulldatainputs.get("filldata_target").get("replace_with_fillna")
        
        for col in columnslist:
            dataf[col]=dataf[col].fillna(np.nan)
    
    
    #"filldata_numerical").get("Removerows_list")
    if handlingnulldatainputs.get("filldata_numerical").get("remove_rows_list")!="":
        columnslist=handlingnulldatainputs.get("filldata_numerical").get("remove_rows_list")
       
        dataf=dataf.dropna(subset=columnslist)
    
    
    #"filldata_numerical").get("Removecolumn_list")
    if handlingnulldatainputs.get("filldata_numerical").get("remove_column_list")!="":
        columnslist=handlingnulldatainputs.get("filldata_numerical").get("remove_column_list")
       
        dataf=dataf.drop(columns=columnslist)
    
    
    #"filldata_categorical").get("Replacewithmode")
    if handlingnulldatainputs.get("filldata_categorical").get("replace_with_mode")!="":
        columnslist=handlingnulldatainputs.get("filldata_categorical").get("replace_with_mode")
      
        for col in columnslist:
            dataf[col]=dataf[col].fillna(dataf[col].mode()[0])
    
   #"filldata_categorical").get("Removecolumn_list")
    if handlingnulldatainputs.get("filldata_categorical").get("remove_column_list")!="":
        columnslist=handlingnulldatainputs.get("filldata_categorical").get("remove_column_list")
      
        dataf=dataf.drop(columns=columnslist)
   
    #"filldata_categorical").get("Removerows_list")
    if handlingnulldatainputs.get("filldata_categorical").get("remove_rows_list")!="":
        columnslist=handlingnulldatainputs.get("filldata_categorical").get("remove_rows_list")
       
        dataf=dataf.dropna(subset=columnslist)
        
    
    #"filldata_target").get("Removerows_list")
    if handlingnulldatainputs.get("filldata_target").get("remove_rows_list")!="":
        columnslist=handlingnulldatainputs.get("filldata_target").get("remove_rows_list")
      
        dataf=dataf.dropna(subset=columnslist)
     
    #"filldata_target").get("Replacewithmode")
    if handlingnulldatainputs.get("filldata_target").get("replace_with_mode")!="":
        columnslist=handlingnulldatainputs.get("filldata_target").get("replace_with_mode")
       
        for col in columnslist:
            dataf[col]=dataf[col].fillna(dataf[col].mode()[0])
                 
    
    
            
    return dataf;
    



# In[ ]:




