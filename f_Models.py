#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns

#Visualization
import matplotlib.pyplot as plt

def main():
    
    datafile=input("Insert data file")
    data = pd.read_csv(datafile)
    x=data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
    y=data['Price']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
    inputmodel=input("log_regression/lin_regression/Clustering:   ")
    
    if inputmodel=="log_regression":
        log_regression(data)
    elif inputmodel=="lin_regression":
        lin_regression(x_train,x_test,y_train,y_test,data,x,y)
    elif inputmodel=="Clustering":
        clustering(data)
        
    
    
def splitfunction(data):
    targetcol=input("Insert target column: ")
    testsize=input("Insert split size: ")
    x=data.drop([targetcol],axis=1)
    y=data[targetcol]
    x_train,x_test,y_train,y_test=train_test_split(data.drop(targetcol,axis=1),data[targetcol],test_size=float(testsize))
    print(x_train.head())
    



def lin_regression(x_train,x_test,y_train,y_test,data,x,y):
   # splitfunction(data)
    lm=LinearRegression()
    lm.fit(x_train,y_train)
    print("intercept:")
    print(lm.intercept_)
    print("coef:")
    print(lm.coef_)
   # cat_vari=data.columns[data.dtypes == 'object']
    cat_vari=['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']
    coeff=pd.DataFrame(lm.coef_,index=[cat_vari])
    print("coeff: ")
    print(coeff)
    predict=lm.predict(x_test)
    print("predict: ")
    print(predict)
    print(plt.scatter(predict,y_test))
    print(sns.displot((y_test-predict),bins=50))
    # print the R-squared value for the model
    print(lm.score(x, y))
    print(metrics.mean_absolute_error(y_test,predict))
    print(metrics.mean_squared_error(y_test,predict))
    print(np.sqrt(metrics.mean_squared_error(y_test,predict)))




    


# In[ ]:





# In[ ]:





# In[ ]:




