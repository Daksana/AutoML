# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:27:55 2022

@author: Daksana
"""

from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify
import os
import pandas as pd
#import category_encoders as ce
import numpy as np # linear algebra
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from pandas_profiling import ProfileReport
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from os.path import join, dirname, realpath
from outlier_removal_v5 import main2
#from f_handling_missing_value import main
from f_handling_missing_value_v2 import main1
from f_Encoding_3 import main3
from feature_selection_v1 import main4
from flask_cors import CORS, cross_origin
import json
import csv
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from model_selection_v1 import main5



app = Flask(__name__)
# enable debugging mode
# app.config["DEBUG"] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/test",methods=["GET", "POST"])
@cross_origin()
def test():
    if request.method == "POST":
        req_data=request.get_json()
        save_path = '/home/senzmate/AutoML/frontEnd'
        file_name = "test.txt"
        completeName = os.path.join(save_path, file_name)
        print(completeName)
        
        file1 = open(completeName, "w")
        file1.write("file information")
        file1.close()
        
        print("filedata_created")
        return {"status":"success"}
    
    return render_template("index.html")

@app.route("/create-model", methods=["GET", "POST"])
@cross_origin()
def create_model():
    if request.method == "POST":
        csv_upload = request.files['file']
        filename = csv_upload.filename
        csv_upload.save(os.path.join("uploads", csv_upload.filename))
        path = os.path.join("uploads", csv_upload.filename)
        df = pd.read_csv(path)
        # prof_report = ProfileReport(df)
        # prof_report.to_file(os.path.join("/home/senzmate/AutoML/html/htmlreport",str(filename)+"_profile_report_new.html"))
        req_data=request.get_json()
        print("reqdata:")
        print(req_data)
        #print(req_data.get("name"))
        
       
        #model_name=req_data["name"]
        # complete_file_data={ModelName:
        #      {
        #      "newModelName":req_data["newModelName"],
        #      "property":req_data["property"],
        #      "device":req_data["device"],
        #      "uploaded_csv_filename":filename,
        #      "uploaded_csv_path":path
        #      }
        #      }
        complete_file_data={
            # "newModelName":req_data["newModelName"],
            # "property":req_data["property"],
            # "device":req_data["device"],
            "uploaded_csv_filename":filename,
            "uploaded_csv_path":path,
            #"model_name":model_name
            }
              
        
       
        #model_name_file_path="/home/senzmate/AutoML/backEnd/"+model_".json"
        # value=os.path.exists(model_name_file_path)
        # if not value:
        #     with open(model_name_file_path, 'w') as fp:
        #         json.dump(complete_file_data, fp)
         
        with open('complete_file_data1.json', 'w') as fp:
            json.dump(complete_file_data, fp)
        output= {"status":"success"}
        
        
        return output;
    return{"status":"failed"};
 
@app.route("/visualization", methods=["GET", "POST"])
@cross_origin()
def visualization():
    if request.method == "GET":
        
        with open('complete_file_data1.json', 'r') as fp:
                complete_file_data = json.load(fp) 
        filename=complete_file_data["uploaded_csv_filename"]
        filepath=complete_file_data["uploaded_csv_path"]
        profile_report_path="/home/senzmate/AutoML/html/index.html"
        complete_file_data["profile_report_path"]=profile_report_path
        
        with open('complete_file_data1.json', 'w') as fp:
            json.dump(complete_file_data, fp)
            
        
        
        df=pd.read_csv(filepath)
        df = df.fillna('')
        #csv data
        csvData = df.to_dict("records")
        
        
        
        #finding null% in columns
        missing_values=df.isnull().sum()
        x=missing_values[missing_values>0]/len(df)*100
        nullvalue_percentage=x.to_dict()
        
        # nullvalue_percentage={
        # "Age": 19.865319865319865,
        # "Cabin": 77.10437710437711,
        # "Embarked": 0.22446689113355783
        # }
        
        output= {"status":"success","path":profile_report_path,"csvData":csvData,"nullvalue_percentage":nullvalue_percentage}
        
        
        return output;
    return{"status":"failed"};
 


@app.route("/handlingnull", methods=["GET", "POST"])
@cross_origin()
def handlingnull():
    if request.method == "POST":
        with open('complete_file_data1.json', 'r') as fp:
                complete_file_data = json.load(fp) 
        req_data=request.get_json()
        filepath=complete_file_data["uploaded_csv_path"]
        print(req_data)
        # print("req_data_hn as dict")
        # print(json.loads(req_data_hn))
        handlingnulldatainputs= {

                                "numerical_column_list":req_data["numerical_column_list"],
                                "categorical_column_list":req_data["categorical_column_list"],
                                "target_column_list":req_data["target_column_list"],
                                
                                "filldata_numerical":{
                                "remove_column_list":req_data["filldata_numerical"]["n_remove_column_list"],
                                "remove_rows_list":req_data["filldata_numerical"]["n_remove_rows_list"],
                                "replace_with_zero":req_data["filldata_numerical"]["n_replace_with_zero"],
                                "replace_with_mean":req_data["filldata_numerical"]["n_replace_with_mean"],
                                "replace_with_median":req_data["filldata_numerical"]["n_replace_with_median"],
                                "replace_with_mode":req_data["filldata_numerical"]["n_replace_with_mode"],
                                "replace_with_fillna":req_data["filldata_numerical"]["n_replace_with_na"]
                                },
        
                                "filldata_categorical":{
                                "remove_column_list":req_data["filldata_categorical"]["c_remove_column_list"],
                                "remove_rows_list":req_data["filldata_categorical"]["c_remove_rows_list"],
                                "replace_with_mode":req_data["filldata_categorical"]["c_replace_with_mode"],
                                "replace_with_fillna":req_data["filldata_categorical"]["c_replace_with_na"]
                                },
                                
                                "filldata_target":{
                                "remove_rows_list":req_data["filldata_target"]["t_remove_rows_list"],
                                "replace_with_mode":req_data["filldata_target"]["t_replace_with_mode"],
                                "replace_with_fillna":req_data["filldata_target"]["t_replace_with_na"]
                                }

                                }
        
        
                
        complete_file_data["numerical_column_list"]=handlingnulldatainputs["numerical_column_list"]
        complete_file_data["categorical_column_list"]=handlingnulldatainputs["categorical_column_list"]
        complete_file_data["target_column_list"]=handlingnulldatainputs["target_column_list"]
        
        numerical_columns=handlingnulldatainputs["numerical_column_list"]
        categorical_columns=handlingnulldatainputs["categorical_column_list"]
        target_column=handlingnulldatainputs["target_column_list"]
        
        
        with open('complete_file_data1.json', 'w') as fp:
            json.dump(complete_file_data, fp)
            
        # print("handlingnulldatainputs")
        ##print(handlingnulldatainputs)
        dataf=pd.read_csv(filepath)
        
        #handling non numeric characters in numerical columns
        for column in numerical_columns:
            dataf[column]=pd.to_numeric(dataf[column],errors='coerce')
            dataf[column]=dataf[column].replace('NaN', np.nan)
        
        #create new ataframe based on user column selection
        
        all_column=numerical_columns+categorical_columns+target_column
        print("all_column")
        print(all_column)
        pre_columns=list(dataf.columns)
        print("pre_columns")
        print(pre_columns)
        set_columns=[]
        for item in pre_columns:
              if item not in all_column:
                set_columns.append(item)
        print("set_columns")
        print(set_columns)
        dataf.drop(set_columns, axis = 1, inplace=True)
        
        nullremoved_dataframe=main1(dataf,handlingnulldatainputs)
        
        #Display columns with missing value with missing records %
        missing_values=nullremoved_dataframe.isnull().sum()
        print("missing_values%:")
        print(missing_values[missing_values>0]/len(nullremoved_dataframe)*100)
        # nullremoved_dataframe_csvfile=nullremoved_dataframe.to_csv("data_after_nullr.csv")
        # nullremoved_dataframe_csvfile.save(os.path.join("working_csvfiles/file1", "data_after_nullr.csv"))
        #nullremoved_dataframe_csv = nullremoved_dataframe.to_dict("records")
        
        ###find column list in nullremoved dataframe
        all_columns_nr=list(nullremoved_dataframe.columns)
        for item in all_column:
             if item not in all_columns_nr:
                if item in numerical_columns:
                    numerical_columns.remove(item)
                if item in categorical_columns:
                    categorical_columns.remove(item)
                if item in target_column:
                    target_column.remove(item)
        #############outlier visialization for each column############
        target_column_1=handlingnulldatainputs["target_column_list"]
        plots_path_dict={}
        
        print("test")
        print(numerical_columns)
        print(categorical_columns)
        direc="/Users/xfach/.ssh/upload_filecode/plots/"
        
        
       
        for col in numerical_columns:
            if col in numerical_columns:
                    sns.set_theme(style="whitegrid")
                    ax = sns.boxplot(x=nullremoved_dataframe[col])
                    plt.savefig(direc+col+"_boxplot.png")
                    key_boxplot="countplot_"+col
                    plots_path_dict[key_boxplot]=direc+col+"_boxplot.png"          
        for col in categorical_columns:
            if col in categorical_columns:
                  sns.set_theme(style="darkgrid")
                  ax = sns.countplot(x=col, data=nullremoved_dataframe)
                  plt.savefig(direc+col+"_countplot.png")
                  key_hist="countplot_"+col
                  plots_path_dict[key_hist]=direc+col+"_countplot.png" 
            
        
        # for col_n in numerical_columns:
        #     sns.boxplot(x =col_n, y =target_column[0] , data = nullremoved_dataframe)
        #     plt.savefig(direc+col_n+"_boxplot.png")
        #     key_boxplot="boxplot_"+col_n
        #     plots_path_dict[key_boxplot]=direc+col_n+"_boxplot.png"
            
            
     
        # for col in numerical_columns:
        #     x = nullremoved_dataframe[target_column_1]
        #     y = nullremoved_dataframe[col]
        #     sns.boxplot(x,y,data = nullremoved_dataframe)
        #     plt.savefig(direc+col+"_boxplot.png")
        #     key_boxplot="boxplot_"+col
        #     plots_path_dict[key_boxplot]=direc+col+"_boxplot.png"
         
        # for col in numerical_columns:
        #     x = nullremoved_dataframe[target_column_1]
        #     y = nullremoved_dataframe[col]
        #     sns.boxplot(x,y,data = nullremoved_dataframe)
        #     plt.savefig(direc+col+"_boxplot.png")
        #     key_boxplot="boxplot_"+col
        #     plots_path_dict[key_boxplot]=direc+col+"_boxplot.png"
    
        # for col in list(nullremoved_dataframe):
        #     direc="/Users/xfach/.ssh/upload_filecode/plots/"
        #     if col in cat_var:
        #         x=nullremoved_dataframe[target_column_1]
        #         y=nullremoved_dataframe[col]
        #         sns.catplot(x,y,kind="box",data = nullremoved_dataframe)
        #         plt.savefig(direc+col+"_boxplot.png")
                
        #     else:
        #         x = nullremoved_dataframe[target_column_1]
        #         y = nullremoved_dataframe[col]
        #         sns.boxplot(x,y,data = nullremoved_dataframe)
        #         plt.savefig(direc+col+"_boxplot.png")

        #         key_boxplot="boxplot_"+col
        #         plots_path_dict[key_boxplot]=direc+col+"_boxplot.png"
                
        
        nullremoved_dataframe.to_csv("data_after_nullremoval.csv",index=False)
               
        return {"status":"success","plots_path_dict":plots_path_dict}
    return render_template("index.html")

@app.route("/outlierremoval", methods=["GET", "POST"])
@cross_origin()
def outlierremoval():
    if request.method == "POST":
        req_data=request.get_json()
        print("reqdata:")
        print(req_data)
        dataf=pd.read_csv("/Users/xfach/.ssh/upload_filecode/data_after_nullremoval.csv")
        #dataf=pd.read_csv("/home/senzmate/AutoML/backEnd/data_after_nullremoval.csv")
        print(dataf)
        outlierremovaldatainputs= {
                                "trimming_column_list":req_data["trimming_column_list"],
                                "fandc_column_list":req_data["fandc_column_list"],
                                } 
                
        outlier_removal_dataframe=main2(dataf,outlierremovaldatainputs)
        outlier_removal_dataframe.to_csv("data_after_outrem.csv",index=False)
        return {"status":"success/send encoding data"}
    return render_template("index.html")

@app.route("/encoding", methods=["GET", "POST"])
@cross_origin()
def encoding():
    if request.method == "POST":
        req_data=request.get_json()
        with open('complete_file_data1.json', 'r') as fp:
                complete_file_data = json.load(fp) 
        uploaded_csv_filename=complete_file_data["uploaded_csv_filename"]
        stripped_csv_filename = uploaded_csv_filename.replace(".csv", "")
        tar_col=complete_file_data["target_column_list"]
        encodingdatainputs= {
                                "labelencoding_column_list":req_data["labelencoding_column_list"],
                                "onehotencoding_column_list":req_data["onehotencoding_column_list"],
                                "minmaxscaler_column_list":req_data["minmaxscaler_column_list"],
                                "standardscaler_column_list":req_data["standardscaler_column_list"]
                                }
        
        dataf=pd.read_csv("/Users/xfach/.ssh/upload_filecode/data_after_outrem.csv")
        encoded_dataframe=main3(dataf,encodingdatainputs,tar_col)
        encoded_dataframe.to_csv("data_after_encoding.csv",index=False)
        encoded_dataframe_csv = encoded_dataframe.to_dict("records")
        
        #report after preprocessing for visualization.
        # report = ProfileReport(encoded_dataframe)
        # report.to_file(os.path.join("/home/senzmate/AutoML/html",str(stripped_csv_filename)+"_profile_report_final.html"))
        # path_r = os.path.join("/home/senzmate/AutoML/html/htmlreport/",str(stripped_csv_filename)+"_profile_report_final.html")
        path="/home/senzmate/AutoML/html/htmlreport/titanic_profile_report_final.html"

        return {"status":"success","encoded_csvdata":encoded_dataframe_csv,"report_after_preprocesing":path}
    
    return render_template("index.html")






@app.route("/featureselection", methods=["GET", "POST"])
@cross_origin()
def featureselection():
    if request.method == "POST":
        req_data=request.get_json()

        featureselectioninputs= {
                    "manual_selection":req_data["manual_selection"],
                    "column_list":req_data["column_list"],
                    "target":req_data["target"],
                    "variance_threshold":req_data["variance_threshold"],
                    "k_best":req_data["k_best"],
                    "recursive_fs":req_data["recursive_fs"]
                               
                                }
        dataf=pd.read_csv("/Users/xfach/.ssh/upload_filecode/data_after_encoding.csv")
        fs_dataframe,alogorithm_data=main4(dataf,featureselectioninputs)
        fs_dataframe.to_csv("data_after_featureselection.csv",index=False)
        fs_dataframe = fs_dataframe.to_dict("records")
        return {"status":"feature selection_success","fs_csvdata":fs_dataframe,"alogorithm_data":alogorithm_data}
    return render_template("index.html")

        
@app.route("/modelselection", methods=["GET", "POST"])
@cross_origin()
def modelselection():
    if request.method == "POST":
        req_data=request.get_json()
        with open('complete_file_data1.json', 'r') as fp:
                complete_file_data = json.load(fp) 
        target_colm=complete_file_data["target_column_list"]
        filename=complete_file_data["uploaded_csv_filename"]
        stripped_csv_filename = filename.replace(".csv", "")
        print(target_colm)
        dataf=pd.read_csv("/Users/xfach/.ssh/upload_filecode/data_after_featureselection.csv")
        
        # modelselectioninputs={                 
        #             "logistic_regression":req_data["logistic_regression"],
        #             "logistic_regression_parameters":req_data["logistic_regression_parameters"],
        #             "knn":req_data["knn"],
        #             "knn_parameters":{
        #             "max_depth":req_data["knn_parameters"]["max_depth"],
        #             "min_samples_split":req_data["knn_parameters"]["min_samples_split"],
        #             "n_estimators":req_data["knn_parameters"]["n_estimators"],
        #             "random_state":req_data["knn_parameters"]["random_state"],
        #             "criterion":req_data["knn_parameters"]["criterion"],
        #             },
        #             "svm":req_data["svm"],
        #             "svm_parameters":req_data["svm_parameters"],
        #             "randomforest":req_data["randomforest"],
        #             "randomforest_parameters":req_data["randomforest_parameters"],
        #             "decisiontree":req_data["decisiontree"],
        #             "decisiontree_parameters":req_data["decisiontree_parameters"]
        #             }

        modelselectioninputs=req_data
        accuracy_score,f1_score,precision,recall,cm_path=main5(dataf,modelselectioninputs,target_colm,stripped_csv_filename)
       
        
        return {"status":"success","accuracy_score":accuracy_score,"f1_score":f1_score,"precision":precision,"recall":recall,"cm_path":cm_path}
        
    return render_template("index.html")

if (__name__ == "__main__"):
    print("running")
    app.run(host="0.0.0.0",port=5000,debug=True)

     
