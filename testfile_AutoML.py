#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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
#from outlier_removal_v5 import main
#from f_handling_missing_value import main
#from f_handling_missing_value_v2 import main1
#from f_Encoding_3 import main
from flask_cors import CORS, cross_origin
import json
import csv
app = Flask(__name__)
@app.route("/test")
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



if (__name__ == "__main__"):
    print("running")
    app.run(host="0.0.0.0",port=5000,debug=True)



