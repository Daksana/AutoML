# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:27:55 2022

@author: Daksana
"""

from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify
import os
import pandas as pd
import numpy as np # linear algebra
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from pandas_profiling import ProfileReport
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from os.path import join, dirname, realpath
from outlier_removal_v4 import main
from f_handling_missing_value import main
#from f_handling_missing_value_v2 import main
from f_Encoding_2 import main
from flask_cors import CORS, cross_origin
import category_encoders as ce
import json
import csv


app = Flask(__name__)
# enable debugging mode
# app.config["DEBUG"] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/upload-csv", methods=["GET", "POST"])
@cross_origin()
def upload_csv():
    
    if request.method == "POST":
        print("request"+str(request))
        print("request.files"+ str(request.files))
        if request.files:
            
            csv_upload = request.files['file']
          #  filename = csv_upload.filename
            csv_upload.save(os.path.join("uploads", csv_upload.filename))

            path = os.path.join("uploads", csv_upload.filename)
            
            df = pd.read_csv(path)
            profile = ProfileReport(df)
            profile.to_file(os.path.join("downloads", "Profile_report.html"))
            

    return render_template("index.html")

@app.route("/create-model", methods=["GET", "POST"])
@cross_origin()
def create_model():
    
    if request.method == "POST":
        # print("request"+str(request))
        # print("request.files"+ str(request.files))
        csv_upload = request.files['file']
        filename = csv_upload.filename
        csv_upload.save(os.path.join("uploads", csv_upload.filename))

        path = os.path.join("uploads", csv_upload.filename)
        # req_data=request.get_json()
        # print("req_data"+str(req_data))
        val=request.values    
        print("val"+str(val))
        req_data=json.loads(val["data"])
        print("req_data"+str(req_data))
        filedata= {
                    "newModelname":req_data["newModelname"],
                    "device":req_data["device"],
                    "name":req_data["name"],
                    "property":req_data["property"]
                }
        
        df = pd.read_csv(path)
        
        print(df.info())
        profile = ProfileReport(df)
        profile.to_file(os.path.join("downloads", str(csv_upload.filename)+"Profile_reportnew.html"))
        profrep_path=os.path.join("downloads/"+str(csv_upload.filename)+"Profile_reportnew.html")
        
        print("report generated")
        modeldata={ 
            "new_modelname":req_data["new_modelname"],
            "device":req_data["device"],
            "name":req_data["name"],
            "property":req_data["property"],
            "csvfilepath":path,
            "profile_report_path":profrep_path    #downloads/titanicProfile_reportnew.html
          
          }
        #convert df to json to be sent to frontend.
        #dfj = df.to_json(orient='index')
        with open('modeldata1.json', 'w') as fp:
            json.dump(modeldata, fp)
        datalist = []
        # with open(path, encoding='utf-8') as csvf:
        #     csvReader = csv.DictReader(csvf)
    
        # for rows in csvReader:
        #     datalist.append(rows)
        #dfj = df.to_dict(orient='list')
        missing_values=df.isnull().sum()
        x=missing_values[missing_values>0]/len(df)*100
        nullvalue_percentage=x.to_dict()
        dfj = df.to_dict("records")
        output= {"status":"success","profile_report_path":profrep_path,"csvdata":dfj,"nullvalue_percentage":nullvalue_percentage}
        print("output:"+str(output))
        return output

    return {"status":"success"}



@app.route("/handlingnull", methods=["GET", "POST"])
@cross_origin()
def handlingnull():
    if request.method == "POST":
        req_data=request.get_json()
        handlingnulldatainputs= {
                                "path":req_data["path"],
                                "numerical_column_list":req_data["numerical_columnlist"],
                                "categorical_column_list":req_data["categorical_columnlist"],
                                "target_column":req_data["target_column"],
                                
                                "fill_data_numerical":{
                                "remove_column_list":req_data["remove_column_list"],
                                "remove_rows_list":req_data["remove_rows_list"],
                                "replace_with_na":req_data["replace_with_na"],
                                "replace_with_zero":req_data["replace_with_zero"],
                                "replace_with_mean":req_data["replace_with_mean"],
                                "replace_with_median":req_data["replace_with_median"],
                                "replace_with_mode":req_data["replace_with_mode"]
                                },
                                
                                "fill_data_categorical":{
                                "remove_column_list":req_data["remove_column_list"],
                                "remove_rows_list":req_data["remove_rows_list"],
                                "replace_with_mode":req_data["replace_with_mode"]
                                },
                                
                                "fill_data_target":{
                                "replace_with_na":req_data["replace_with_na"],
                                "remove_rows_list":req_data["remove_rows_list"],
                                "replace_with_mode":req_data["replace_with_mode"]
                                }

                                }
        dataf=pd.read_csv(handlingnulldatainputs["path"])
        nullremoved_dataframe=main(dataf,handlingnulldatainputs)
        nullremoved_dataframe.to_csv("data_after_nullr.csv")
        return {"status":"success"}
    return render_template("index.html")

@app.route("/outlierremoval", methods=["GET", "POST"])
@cross_origin()
def outlierremoval():
    print("outlierremoval")
    if request.method == "POST":
     #   df = pd.read_csv(path)
        with open('modeldata1.json', 'r') as fp:
                modeldata = json.load(fp) 
        req_data=request.get_json()
        outlierremovaldatainputs= {
                                "columnname":req_data["columnname"],
                                "dataretrival_method":req_data["dataretrival_method"],
                                "removalmethod":req_data["removalmethod"]
                                }
        #to set name of file in vm
        #path="/uploads/"
        path=modeldata["csvfilepath"]
        dataf=pd.read_csv(path)
        print("data printed...............................")
        print(outlierremovaldatainputs.get("columnname"))
        outlier_removal_dataframe=main(dataf,outlierremovaldatainputs.get("columnname"),outlierremovaldatainputs.get("removalmethod"))
        outlier_removal_dataframe.to_csv("data_after_or.csv")
        return {"status":"success"}
    return render_template("index.html")

@app.route("/encoding", methods=["GET", "POST"])
@cross_origin()
def encoding():
    if request.method == "POST":
        req_data=request.get_json()
        encodingdatainputs= {
                                "path":req_data["path"],
                                "encodingdecision":req_data["encodingdecision"],
                                "columnname":req_data["columnname"],
                                }
        dataf=pd.read_csv(encodingdatainputs["path"])
        encoded_dataframe=main(dataf,encodingdatainputs.get("encodingdecision"),encodingdatainputs.get("columnname"))
        encoded_dataframe.to_csv("data_after_encoding.csv")
        return {"status":"success"}
    return render_template("index.html")

# @app.route("/vis_after_prepro", methods=["GET", "POST"])
# def visualization():
#     if request.method == "GET":
#         reportpath="C:\Users\xfach\Senzmate\upload_filecode\downloads"   
#         return {"status":"success",
#                 "path":"reportpath"}
#     return render_template("index.html")



@app.route("/featureselection", methods=["GET", "POST"])
@cross_origin()
def featureselection():
    if request.method == "POST":
        req_data=request.get_json()
        featureselectioninputs= {
                                "path":req_data["path"],
                                "columnnames":req_data["columnname"],
                                }
        
        return {"status":"success"}
    return render_template("index.html")

# @app.route("/Preprocessing", methods=["GET", "POST"])
# def Preprocessing(df):

#     if request.method == "POST":
#         out.outliervisualoneway(df)
        
@app.route("/modelselection", methods=["GET", "POST"])
@cross_origin()
def modelselection():
    if request.method == "POST":
        req_data=request.get_json()
        modelselectioninputs= {
                                
                                }
        
        return {"status":"success"}
    return render_template("index.html")

if (__name__ == "__main__"):
    print("running")
    app.run(host="0.0.0.0",port=5000,debug=True)

     
