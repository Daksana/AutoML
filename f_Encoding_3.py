#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
#import category_encoders as ce
#Visualization
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
def main3(dataf,encodingdatainputs,tar_col):
    
    data = dataf
    
    # ord_enc = OrdinalEncoder()
    # data[tar_colmn] = ord_enc.fit_transform(data[[tar_colmn]])
    # cat_var_col=encodingdatainputs.get("cat_column_list")
    # num_var_col=encodingdatainputs.get("num_column_list")
    labelencoding_col=encodingdatainputs.get("labelencoding_column_list")
    onehotencoding_col=encodingdatainputs.get("onehotencoding_column_list")
    minmaxscaler_col=encodingdatainputs.get("minmaxscaler_column_list")
    standardscaler_col=encodingdatainputs.get("standardscaler_column_list")
    
    print(data.columns)
    for col in labelencoding_col:
        
        label_encoder = preprocessing.LabelEncoder()
        data[col]= label_encoder.fit_transform(data[col])

    print(data.columns)
    for col in onehotencoding_col:
        data = pd.get_dummies(data, columns=[col], prefix = [col])
        
    print(data.columns)
    for col in standardscaler_col:
        X = np.array(data[col]).reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        data[col] = X_scaled.reshape(1,-1)[0]
        
    print(data.columns)   
    for col in minmaxscaler_col:
        X = np.array(data[col]).reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        data[col] = X_scaled.reshape(1,-1)[0]
    
    
    
    return data #display enocoded data frame.



# In[ ]:




