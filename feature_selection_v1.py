#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify
import os
import pandas as pd
from os.path import join, dirname, realpath
from flask_cors import CORS, cross_origin
import json
import csv

def main4(dataf,featureselectioninputs):
    
    data = dataf
    problem_type=featureselectioninputs.get("problem_type")
    
    # print("manual_fs_columns")
    # print(featureselectioninputs.get("column_list"))
    # print("columns")
    # print(data.columns.tolist())
    # columns=data.columns.tolist()
    # res = [ ele for ele in columns ]
    # print("res")
    # print(res)
    # data1=data
     
    if featureselectioninputs["manual_selection"]:
        if problem_type=="classification" or problem_type=="regression":
            targetcol=featureselectioninputs.get("target")
            print("type")
            print(type(targetcol))
            manual_fs_columns=featureselectioninputs.get("column_list")
            columns=data.columns.tolist()
            
            res = [ ele for ele in columns ]
            for a in manual_fs_columns:
              if a in columns:
                res.remove(a)
            res.remove(targetcol)
            
            data1=data.drop(res, axis = 1)
        if problem_type=="clustering":
            manual_fs_columns=featureselectioninputs.get("column_list")
            columns=data.columns.tolist()
            
            res = [ ele for ele in columns ]
            for a in manual_fs_columns:
              if a in columns:
                res.remove(a)
            data1=data.drop(res, axis = 1)
        
    if featureselectioninputs["variance_threshold"]:
        mdlsel = VarianceThreshold(threshold=0.5)
        mdlsel.fit(data)
        ix = mdlsel.get_support()
        data1 = mdlsel.transform(data) 
        data1 = pd.DataFrame(mdlsel.transform(data), columns = data.columns.values[ix])
        
    if problem_type=="classification":
        algorithm_data=[
      {
        "algo_name":"Logistic Regression",
        "parameters":[]
      },
      {
        "algo_name":"Naive Bayes",
        "parameters":[
          {
            "name":"estimator",
            "type":"dropdown",
            "options":["GaussianNB()","MultinomialNB()","BernoulliNB()"],
            "value":"GaussianNB()"
          },
           {
            "name":"verbose",
            "type":"number",
            "value":1
          },
           {
            "name":"cv",
            "type":"number",
            "value":10
          },
          {
            "name":"n_jobs",
            "type":"number",
            "value":-1
          }
        ]
      },
      {
        "algo_name":"RandomForestClassifier",
        "parameters":[
          {
            "name":"max_depth",
            "type":"number",
            "value":"None"
          },
           {
            "name":"n_estimators",
            "type":"number",
            "value":100
          },
           {
            "name":"random_state",
            "type":"number",
            "value":"None"
          },
          {
            "name":"min_samples_split",
            "type":"number",
            "value":2
          },
          {
            "name":"criterion",
            "type":"dropdown",
            "options":["gini","entropy"],
            "value":"gini"
          }
        ]
      },
      {
        "algo_name":"SVM",
        "parameters":[
         {
            "name":"C",
            "type":"number",
            "value":1.0
          },
          {
            "name":"gamma",
            "type":"dropdown",
            "options":["scale","auto"],
            "value":"scale"
          },
          {                                          
            "name":"kernel",
            "type":"dropdown",
            "options":["linear","poly","rbf","sigmoid","precomputed"],
            "value":"rbf"
          },
          {
            "name":"degree",
            "type":"number",
            "value":3
          }
    
        ]
      },
       {
        "algo_name":"Decisiontree",
        "parameters":[
          {
            "name":"max_depth",
            "type":"number",
            "value":"None"
          },
          {
            "name":"criterion",
            "type":"dropdown",
            "options":["gini","entropy"],
            "value":"gini"
          },
          {
            "name":"splitter",
            "type":"dropdown",
            "options":["best","random"],
            "value":"best"
          },
          {
            "name":"min_samples_split",
            "type":"number",
            "value":2
          }
        ]
      }
       
       
    ]
        
    
    if problem_type=="regression":
        algorithm_data=[
      {
        "algo_name":"Linear Regression",
        "parameters":[
    
          {
            "name":"copy_X",
            "type":"dropdown",
            "options":["True","False"],
            "value":"True"
          },
          {
            "name":"fit_intercept",
            "type":"dropdown",
            "options":["True","False"],
            "value":"True"
          },
          {
            "name":"n_jobs",
            "type":"number",
            "value":1
          },
          {
            "name":"normalize",
            "type":"dropdown",
            "options":["True","False"],
            "value":"True"
          }
        ]
      },
       {
        "algo_name":"Multiple Regression",
        "parameters":[
    
          {
            "name":"copy_X",
            "type":"dropdown",
            "options":["True","False"],
            "value":"True"
          },
          {
            "name":"fit_intercept",
            "type":"dropdown",
            "options":["True","False"],
            "value":"True"
          },
          {
            "name":"n_jobs",
            "type":"number",
            "value":1
          },
          {
            "name":"normalize",
            "type":"dropdown",
            "options":["True","False"],
            "value":"True"
          }
        ]
      },
       {
        "algo_name":"Ridge Regression",
        "parameters":[
          {
            "name":"alpha",
            "type":"number",
            "value":0.01
          }
        ]
      },
       {
        "algo_name":"Lasso Regression",
        "parameters":[
          {
            "name":"alpha",
            "type":"number",
            "value":0.01
          }
        ]
      }
    ]
        
    if problem_type=="clustering":
        algorithm_data=[
      {
      
        "algo_name":"K-means",
        "parameters":[
          {
            "name":"n_clusters",
            "type":"dropdown",
            "options":["default","2","3","4","5","6","7","8"],
            "value":"default"
          },
           {
            "name":"max_iter",
            "type":"number",
            "value":50
          },
           {
            "name":"random_state",
            "type":"number",
            "value":50
          }
         
        ]
      }
     
       
    ]
    return data1,algorithm_data;

