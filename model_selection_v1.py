

import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
import seaborn as sns
#import category_encoders as ce
#Visualization
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import ast
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans

# To perform PCA
from sklearn.decomposition import PCA
#To perform hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


def main5(data,modelselectioninputs,target_column,stripped_csv_filename):
    df = data
    target=target_column
    filename=stripped_csv_filename
    problem_type=modelselectioninputs["problem_type"]
    
    if problem_type=="classification":
    
        if modelselectioninputs["algo_name"]=="Logistic Regression":
            a_model_name=modelselectioninputs["algo_name"]
            target = df[target_column]
            features=df
            features.drop(target, inplace=True, axis=1)
        
            X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            
            
            classifier = linear_model.LogisticRegression()
            classifier_ = classifier.fit(X_train,y_train)
            target_predict=classifier_.predict(X_test)
            
            
            precision_score_value=precision_score(y_test, target_predict)
            recall_score_value=recall_score(y_test, target_predict)
            accuracy_score_value=accuracy_score(y_test,target_predict)
            f1_score_value =f1_score(y_test,target_predict)
            plot_confusion_matrix(classifier_, X_test, y_test) 
            #directory="C:/Users/xfach/.ssh/upload_filecode/"
            directory="/home/senzmate/AutoML/html/confusionmatrix/"
            plt.savefig(directory+filename+a_model_name+"_confusionmatrix.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_confusionmatrix.png"
            return accuracy_score_value,f1_score_value,precision_score_value,recall_score_value,cm_path;
    
        if modelselectioninputs["algo_name"]=="RandomForestClassifier":
            a_model_name=modelselectioninputs["algo_name"]
            target = df[target_column].values
            print(target)
            df.drop(target_column, inplace=True, axis=1)
            features=df.values
            print(features)
            
        
            X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            print("splitted")
            
            #getting parameter values from user input.
            
            parameters=(modelselectioninputs.get("parameters"))
    
            for i in parameters:
                for x in i:
                    key=i[x]
                    value=key["value"]
                    if x=="max_depth":
                        if value=="None":
                            newvalue=ast.literal_eval(value)
                            max_depthv=newvalue
                        else:
                            max_depthv=value
                    if x=="min_samples_split":
                        min_samples_splitv=value
                    if x=="random_state":
                        if value=="None":
                            newvalue=ast.literal_eval(value)
                            random_statev=newvalue
                        else:
                            random_statev=value
                    if x=="n_estimators":
                        n_estimatorsv=value
                    if x=="criterion":
                        criterionv=value
        
            print("recieved input")
            
            my_forest = RandomForestClassifier(max_depth=max_depthv, min_samples_split=min_samples_splitv, n_estimators=n_estimatorsv, random_state=random_statev,criterion = criterionv)
            my_forest_ = my_forest.fit(X_train,y_train)
            target_predict=my_forest_.predict(X_test)
            
            print("trained")
            precision_score_value=precision_score(y_test, target_predict)
            recall_score_value=recall_score(y_test, target_predict)
            accuracy_score_value=accuracy_score(y_test,target_predict)
            f1_score_value =f1_score(y_test,target_predict)
            plot_confusion_matrix(my_forest_, X_test, y_test) 
            #directory="C:/Users/xfach/.ssh/upload_filecode/confusionmatrix/"
            directory="/home/senzmate/AutoML/html/confusionmatrix/"
            plt.savefig(directory+filename+a_model_name+"_confusionmatrix.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_confusionmatrix.png"
            return accuracy_score_value,f1_score_value,precision_score_value,recall_score_value,cm_path;
        
        if modelselectioninputs["algo_name"]=="Decisiontree":
            a_model_name=modelselectioninputs["algo_name"]
            target = df[target_column].values
            print(target)
            df.drop(target_column, inplace=True, axis=1)
            features=df.values
            print(features)
            
        
            X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            print("splitted")
            
            #getting parameter values from user input.
            
            parameters_dt=(modelselectioninputs.get("parameters"))
            
            
            for i in parameters_dt:
                for x in i:
                    key=i[x]
                    value=key["value"]
                    if x=="max_depth":
                        if value=="None":
                            newvalue=ast.literal_eval(value)
                            max_depthv=newvalue
                        else:
                            max_depthv=value
                    if x=="criterion":
                        criterionv=value
                        print(criterionv)
                    if x=="min_samples_split":
                        min_samples_splitv=value
                        print(min_samples_splitv)
                    if x=="splitter":
                        splitterv=value
                        print(splitterv)
            print("recieved input")
            
            
    
            decision_tree = tree.DecisionTreeClassifier(max_depth=max_depthv,criterion =criterionv,min_samples_split = min_samples_splitv,splitter=splitterv)
            
            
            decision_tree_ = decision_tree.fit(X_train,y_train)
            target_predict=decision_tree_.predict(X_test)
    
            
            print("trained")
            precision_score_value=precision_score(y_test, target_predict)
            recall_score_value=recall_score(y_test, target_predict)
            accuracy_score_value=accuracy_score(y_test,target_predict)
            f1_score_value =f1_score(y_test,target_predict)
            plot_confusion_matrix(decision_tree_, X_test, y_test) 
            #directory="C:/Users/xfach/.ssh/upload_filecode/confusionmatrix/"
            directory="/home/senzmate/AutoML/html/confusionmatrix/"
            plt.savefig(directory+filename+a_model_name+"_confusionmatrix.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_confusionmatrix.png"
            
            # accuracy_score_value=3
            # f1_score_value=4
            # precision_score_value=2
            # recall_score_value=1
            # cm_path="way"
            return accuracy_score_value,f1_score_value,precision_score_value,recall_score_value,cm_path;
        
        
        if modelselectioninputs["algo_name"]=="SVM":
            a_model_name=modelselectioninputs["algo_name"]
            target = df[target_column].values
            print(target)
            df.drop(target_column, inplace=True, axis=1)
            features=df.values
            print(features)
            
            
            X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            print("splitted")
            
            #getting parameter values from user input.
            
            parameters_dt=(modelselectioninputs.get("parameters"))
            
            
            for i in parameters_dt:
                for x in i:
                    key=i[x]
                    value=key["value"]
                    if x=="C":
                        Cv=value
                        print(Cv)
                    if x=="gamma":
                        gammav=value
                        print(gammav)
                    if x=="kernel":
                        kernelv=value
                        print(kernelv)
                    if x=="degree":
                        degreev=value
                        print(degreev)
                        
            print("recieved input")
                    
           
            clf = svm.SVC(C=Cv,gamma =gammav,kernel = kernelv,degree=degreev)
            
            
            clf_ = clf.fit(X_train,y_train)
            target_predict=clf_.predict(X_test)
    
            
            print("trained")
            precision_score_value=precision_score(y_test, target_predict)
            recall_score_value=recall_score(y_test, target_predict)
            accuracy_score_value=accuracy_score(y_test,target_predict)
            f1_score_value =f1_score(y_test,target_predict)
            plot_confusion_matrix(clf_, X_test, y_test) 
            #directory="C:/Users/xfach/.ssh/upload_filecode/confusionmatrix/"
            directory="/home/senzmate/AutoML/html/confusionmatrix/"
            plt.savefig(directory+filename+a_model_name+"_confusionmatrix.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_confusionmatrix.png"
            
            # accuracy_score_value=3
            # f1_score_value=4
            # precision_score_value=2
            # recall_score_value=1
            # cm_path="way"
            return accuracy_score_value,f1_score_value,precision_score_value,recall_score_value,cm_path;
        
        if modelselectioninputs["algo_name"]=="Naive Bayes":
            a_model_name=modelselectioninputs["algo_name"]
            target = df[target_column].values
            print(target)
            df.drop(target_column, inplace=True, axis=1)
            features=df.values
            print(features)
            
            
            X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            print("splitted")
            
            #getting parameter values from user input.
            
            parameters_dt=(modelselectioninputs.get("parameters"))
            
            
            for i in parameters_dt:
                for x in i:
                    key=i[x]
                    value=key["value"]
                    if x=="estimator":
                        estimatorv=value
                        print("estimator")
                        print(estimatorv)
                    if x=="verbose":
                        verbosev=value
                    if x=="cv":
                        cvv=value
                    if x=="n_jobs":
                        n_jobsv=value
                   
            
            param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
            if estimatorv=="GaussianNB()":
              nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=verbosev, cv=cvv, n_jobs=n_jobsv)
            if estimatorv=="MultinomialNB()":
              nbModel_grid = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid_nb, verbose=verbosev, cv=cvv, n_jobs=n_jobsv)
            if estimatorv=="BernoulliNB()":
              nbModel_grid = GridSearchCV(estimator=BernoulliNB(), param_grid=param_grid_nb, verbose=verbosev, cv=cvv, n_jobs=n_jobsv)
           
                    
           
            nbModel_grid.fit(X_train, y_train)
            target_predict = nbModel_grid.predict(X_test)
    
            
            print("trained")
            precision_score_value=precision_score(y_test, target_predict)
            recall_score_value=recall_score(y_test, target_predict)
            accuracy_score_value=accuracy_score(y_test,target_predict)
            f1_score_value =f1_score(y_test,target_predict)
            plot_confusion_matrix(nbModel_grid, X_test, y_test) 
            #directory="C:/Users/xfach/.ssh/upload_filecode/confusionmatrix/"
            directory="/home/senzmate/AutoML/html/confusionmatrix/"
            plt.savefig(directory+filename+a_model_name+"_confusionmatrix.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_confusionmatrix.png"
            print(cm_path)
            # accuracy_score_value=3
            # f1_score_value=4
            # precision_score_value=2
            # recall_score_value=1
            # cm_path="way"
            return accuracy_score_value,f1_score_value,precision_score_value,recall_score_value,cm_path;
        
    if problem_type=="regression":
        
        if modelselectioninputs["algo_name"]=="Linear Regression":
            a_model_name=modelselectioninputs["algo_name"]
            target = df[target_column].values
            print(target)
            df.drop(target_column, inplace=True, axis=1)
            features=df.values
            print(features)
            
        
            X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            print("splitted")
            
            #getting parameter values from user input.
            
            parameters_dt=(modelselectioninputs.get("parameters"))
            
            
            for i in parameters_dt:
                for x in i:
                    key=i[x]
                    value=key["value"]
                    if x=="copy_X":
                        value1=json.loads(value.lower())
                        copy_Xv=value1
                    if x=="fit_intercept":
                        value1=json.loads(value.lower())
                        fit_interceptv=value1
                    if x=="n_jobs":
                        n_jobsv=value
                    if x=="normalize":
                        value1=json.loads(value.lower())
                        normalizev=value1
            print("recieved input")
            
            reg=LinearRegression(copy_X=copy_Xv,fit_intercept =fit_interceptv,n_jobs = n_jobsv,normalize=normalizev)
            model = reg.fit(X_train,y_train)
            target_predict = model.predict(X_test)

    
            mae=mean_absolute_error(y_test, target_predict)
            mse=mean_squared_error(y_test, target_predict)
            rmse=np.sqrt(mean_squared_error(y_test, target_predict))
            r2=r2_score(y_test, target_predict)
            
            
            # Set the figure size
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            # Scatter plot
            plt.scatter(y_test,target_predict)
            directory="/home/senzmate/AutoML/html/regressionplot/"
            plt.savefig(directory+filename+a_model_name+"_regressionplot.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_regressionplot.png"
                        
            #directory="C:/Users/xfach/.ssh/upload_filecode/confusionmatrix/"
            
            
            
            
            return mae,mse,rmse,r2,cm_path;
        if modelselectioninputs["algo_name"]=="Multiple Regression":
            a_model_name=modelselectioninputs["algo_name"]
            target = df[target_column].values
            print(type(target))
            df.drop(target_column, inplace=True, axis=1)
            features=df.values
            print(type(features))
           
        
            X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            print("splitted")
            
            #getting parameter values from user input.
            
            parameters_dt=(modelselectioninputs.get("parameters"))
            
            
            for i in parameters_dt:
                for x in i:
                    key=i[x]
                    value=key["value"]
                    if x=="copy_X":
                        value1=json.loads(value.lower())
                        copy_Xv=value1
                    if x=="fit_intercept":
                        value1=json.loads(value.lower())
                        fit_interceptv=value1
                    if x=="n_jobs":
                        n_jobsv=value
                    if x=="normalize":
                        value1=json.loads(value.lower())
                        normalizev=value1
            print("recieved input")
            
            reg=LinearRegression(copy_X=copy_Xv,fit_intercept =fit_interceptv,n_jobs = n_jobsv,normalize=normalizev)
            model = reg.fit(X_train,y_train)
            target_predict = model.predict(X_test)

            print("y_test")
            print(type(y_test))
            print(y_test)
            print("target_predict")
            print(target_predict)
            print(type(target_predict))
            mae=mean_absolute_error(y_test, target_predict)
            mse=mean_squared_error(y_test, target_predict)
            rmse=np.sqrt(mean_squared_error(y_test, target_predict))
            r2=r2_score(y_test, target_predict)
            
            #directory="C:/Users/xfach/.ssh/upload_filecode/confusionmatrix/"
            

            # Set the figure size
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            # Scatter plot
            plt.scatter(y_test,target_predict)
            directory="/home/senzmate/AutoML/html/regressionplot/"
            plt.savefig(directory+filename+a_model_name+"_regressionplot.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_regressionplot.png"
            
            
            #cm_path=""
            return mae,mse,rmse,r2,cm_path;
        if modelselectioninputs["algo_name"]=="Ridge Regression":
            a_model_name=modelselectioninputs["algo_name"]
            target = df[target_column].values
            print(target)
            df.drop(target_column, inplace=True, axis=1)
            features=df.values
            print(features)
            
        
            X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            print("splitted")
            
            #getting parameter values from user input.
            
            parameters_dt=(modelselectioninputs.get("parameters"))
            
            
            for i in parameters_dt:
                for x in i:
                    key=i[x]
                    value=key["value"]
                    if x=="alpha":
                        alphav=value
                    
            
            rr = Ridge(alpha=0.01)
            rr.fit(X_train, y_train) 
            target_predict= rr.predict(X_test)

    
            mae=mean_absolute_error(y_test, target_predict)
            mse=mean_squared_error(y_test, target_predict)
            rmse=np.sqrt(mean_squared_error(y_test, target_predict))
            r2=r2_score(y_test, target_predict)
            
            #directory="C:/Users/xfach/.ssh/upload_filecode/confusionmatrix/"
            # Set the figure size
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            # Scatter plot
            plt.scatter(y_test,target_predict)
            directory="/home/senzmate/AutoML/html/regressionplot/"
            plt.savefig(directory+filename+a_model_name+"_regressionplot.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_regressionplot.png"
            #cm_path=""
            
            
            return mae,mse,rmse,r2,cm_path;
        if modelselectioninputs["algo_name"]=="Lasso Regression":
            a_model_name=modelselectioninputs["algo_name"]
            target = df[target_column].values
            print(target)
            df.drop(target_column, inplace=True, axis=1)
            features=df.values
            print(features)
            
        
            X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
            print("splitted")
            
            #getting parameter values from user input.
            
            parameters_dt=(modelselectioninputs.get("parameters"))
            
            
            for i in parameters_dt:
                for x in i:
                    key=i[x]
                    value=key["value"]
                    if x=="alpha":
                        alphav=value
                    
            
            model_lasso = Lasso(alpha=0.01)
            model_lasso.fit(X_train, y_train) 
            target_predict= model_lasso.predict(X_test)
    
            mae=mean_absolute_error(y_test, target_predict)
            mse=mean_squared_error(y_test, target_predict)
            rmse=np.sqrt(mean_squared_error(y_test, target_predict))
            r2=r2_score(y_test, target_predict)
            
            #directory="C:/Users/xfach/.ssh/upload_filecode/confusionmatrix/"
            # Set the figure size
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            # Scatter plot
            plt.scatter(y_test,target_predict)
            directory="/home/senzmate/AutoML/html/regressionplot/"
            plt.savefig(directory+filename+a_model_name+"_regressionplot.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_regressionplot.png"
            #cm_path=""
            
            
            return mae,mse,rmse,r2,cm_path;
        
    if problem_type=="clustering":
        if modelselectioninputs["algo_name"]=="K-means":
            a_model_name=modelselectioninputs["algo_name"]
            pca_final = IncrementalPCA(n_components=2)
            df_pca = pca_final.fit_transform(df)
            pc = np.transpose(df_pca)
            corrmat = np.corrcoef(pc)
            pcs_df2 = pd.DataFrame({'PC1':pc[0],'PC2':pc[1]})

            model_name=modelselectioninputs["algo_name"]
            
            parameters_dt=(modelselectioninputs.get("parameters"))
            
            
            
            for i in parameters_dt:
                for x in i:
                    key=i[x]
                    value=key["value"]
                    if x=="max_iter":
                        max_iterv=value
                    if x=="random_state":
                        random_statev=value
                    if x=="n_clusters":
                        if value=="default":
                           dat3_1 = pcs_df2
                           range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
                           scorelist = []
                           dict_score={}
                           for num_clusters in range_n_clusters:
                                kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
                                kmeans.fit(dat3_1)
                                
                                cluster_labels = kmeans.labels_
                                
                                 # silhouette score
                                silhouette_avg = metrics.silhouette_score(dat3_1, cluster_labels)
                                scorelist.append(silhouette_avg)
                                dict_score[silhouette_avg]=num_clusters 
                                
                           max_c=max(scorelist)
                           n_opt_cluster=dict_score[max_c]
                           n_clustersv=n_opt_cluster
                        else:
                            n_clustersv=int(value)
                    
            print("recieved input")
            
            my_model = KMeans(n_clusters = n_clustersv, max_iter=max_iterv,random_state = random_statev).fit(df)
            labels = my_model.labels_
            silhouette_score=metrics.silhouette_score(df,labels)
            calinski_harabasz_score=metrics.calinski_harabasz_score(df, labels)
            davies_bouldin_score=metrics.davies_bouldin_score(df,labels)
            
            #directory="C:/Users/xfach/.ssh/upload_filecode/confusionmatrix/"
            
            dat4=pcs_df2
            dat4.index = pd.RangeIndex(len(dat4.index))
            dat_km = pd.concat([dat4, pd.Series(my_model.labels_)], axis=1)
            dat_km.columns = ['PC1', 'PC2','ClusterID']
            fig = plt.figure(figsize = (12,8))
            sns.scatterplot(x='PC1',y='PC2',hue='ClusterID',legend='full',data=dat_km)
            plt.title('Categories')
            directory="/home/senzmate/AutoML/html/clusteringplot/"
            plt.savefig(directory+filename+a_model_name+"_clusteringplot.png")
            plt.clf()
            cm_path=directory+filename+a_model_name+"_clusteringplot.png"
            
            
            #cm_path=""
            return silhouette_score,calinski_harabasz_score,davies_bouldin_score,cm_path;

# In[ ]:




