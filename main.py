#!/usr/bin/env python
# coding: utf-8

# # Advanced Data Mining Final Project 
# ## by: Sohrab Pirhadi 
# ### Student Number: 984112
# 

# # Problem Statement:
# 
# Find a Machine Learning (ML) model that accurately predicts breast cancer based on the 30 features described below.

# 
# # Attribute Information:
# 
# 1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# * a) radius (mean of distances from center to points on the perimeter) 
# * b) texture (standard deviation of gray-scale values) 
# * c) perimeter 
# * d) area 
# * e) smoothness (local variation in radius lengths) 
# * f) compactness (perimeter^2 / area - 1.0) 
# * g) concavity (severity of concave portions of the contour) 
# * h) concave points (number of concave portions of the contour) 
# * i) symmetry 
# * j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 304 benign, 180 malignant

# # Abstract:
#  
#   
#   The Data was split into 80% (~387 people) training and 20% testing (~97 people).
#   
#   Several different models were evaluated through k-fold Cross-Validation with GridSearchCV, which iterates on different algorithm's hyperparameters:
#    * Guassian Naive Bays
#    * Decision Tree
#    * Random Forest
#    * Support Vector Machine
# 
# 
# 

# # Import Libraries 

# In[1]:


import warnings
import os # Get Current Directory
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd # data processing, CSV file I/O (e.i. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from scipy import stats
import subprocess
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# ## Hide Warnings

# In[2]:


warnings.filterwarnings("ignore")
pd.set_option('mode.chained_assignment', None)


# ## Get Current Directory

# In[3]:


currentDirectory=os.getcwd()
print(currentDirectory)


# # Import and View Data

# In[4]:


data= pd.read_csv('wdbc_train.data', sep=',', header = None , names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",])


# In[5]:


data.head(10) # view the first 10 columns


# In[6]:


data.shape # Printing the dimensions of data


# In[7]:


data.columns # Viewing the column heading


# In[8]:


data.diagnosis.value_counts() # Inspecting the target variable


# In[9]:


data.dtypes


# In[10]:


data.nunique() # Identifying the unique number of values in the dataset


# ## Check for Missing Values
# 
# Checking if any NULL values are present in the dataset.no missing values should be present.The following verifies that.

# In[11]:


data.isnull().sum() 


# In[12]:


# See rows with missing values
data[data.isnull().any(axis=1)]


# In[13]:


# Viewing the data statistics
data.describe()


# In[14]:


# Convert Diagnosis for Cancer from Categorical Variable to Binary
diagnosis_num={'B':0,'M':1}
data['diagnosis']=data['diagnosis'].map(diagnosis_num)


# In[15]:


# Verify Data Changes, look at first 5 rows 
data.head(5)


# # Data Visualization

# In[16]:


# Finding out the correlation between the features
corr = data.corr()
corr.shape


# ## Heatmap with Pearson Correlation Coefficient  for Features
# A strong correlation is indicated by a Pearson Correlation Coefficient value near 1.  Therefore, when looking at the Heatmap, we want to see what correlates most with the first column, "diagnosis."  It appears that the features of "concave points worst" [0.8] has the strongest correlation with "diagnosis".  

# In[17]:


fix,ax = plt.subplots(figsize=(22,22))
heatmap_data = data.drop(['id'],axis=1)
sns.heatmap(heatmap_data.corr(),vmax=1,linewidths=0.01,square=True,annot=True,linecolor="white")
bottom,top=ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
heatmap_title='Figure 1:  Heatmap with Pearson Correlation Coefficient for Features'
ax.set_title(heatmap_title)
plt.savefig('Figure1.Heatmap.png',dpi=300,bbox_inches='tight')
plt.show()


# The above heatmap shows us a correlation between the various features. The closer the value to 1, the higher is the correlation between the pair of features.

# ## Analyzing the target variable
# 

# In[18]:


plt.title('Count of cancer type')
sns.countplot(data['diagnosis'])
plt.xlabel('Cancer lethality')
plt.ylabel('Count')
plt.show()


# ### Plotting correlation between diagnosis and radius
# 

# In[19]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(x="diagnosis", y="radius_mean", data=data)
plt.subplot(1,2,2)
sns.violinplot(x="diagnosis", y="radius_mean", data=data)
plt.show()


# Boxplot shows us the minimum, first quartile (Q1), median, third quartile (Q3), and maximum. It is useful for detecting the outliers.
# Violin plot shows us the kernel density estimate on each side.

# ### Plotting correlation between diagnosis and concativity
# 

# In[20]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(x="diagnosis", y="concavity_mean", data=data)
plt.subplot(1,2,2)
sns.violinplot(x="diagnosis", y="concavity_mean", data=data)
plt.show()


# ### Distribution density plot KDE (kernel density estimate)

# In[21]:


sns.FacetGrid(data, hue="diagnosis", height=6).map(sns.kdeplot, "radius_mean").add_legend()
plt.show()


# ### Plotting the distribution of the mean radius
# 

# In[22]:


sns.stripplot(x="diagnosis", y="radius_mean", data=data, jitter=True, edgecolor="gray")
plt.show()


# ### Plotting bivariate relations between each pair of features (4 features x4 so 16 graphs) with hue = "diagnosis"
# 

# In[23]:


sns.pairplot(data, hue="diagnosis", vars = ["radius_mean", "concavity_mean", "smoothness_mean", "texture_mean"])
plt.show()


# Once the data is cleaned, we split the data into training set and test set to prepare it for our machine learning model in a suitable proportion.
# 

# # Split Data for Training  

# ## Standardize and Split the Data

# ### Spliting target variable and independent variables
# 

# In[24]:


X = data.drop(['id','diagnosis'], axis= 1)
y = data.diagnosis


# ### Standardize Data
# 

# In[25]:


scaler = StandardScaler()
X=scaler.fit_transform(X.values)
X = pd.DataFrame(X)
X.columns=(data.drop(['id','diagnosis'], axis= 1)).columns


# A good rule of thumb is to hold out 20 percent of the data for testing. 

# ### Splitting the data into training set and testset
# 

# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

#Fit on training set only.
scaler.fit(X_train)

#Apply transform to both the training and test set
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# ### Verify the Split

# In[27]:


print('X_train - length:',len(X_train), 'y_train - length:',len(y_train))
print('X_test - length:',len(X_test),'y_test - length:',len(y_test))
print('Percent heldout for testing:', round(100*(len(X_test)/len(data)),0),'%')


# # Feature Extraction with PCA

# ##  Feature Extraction:  Principal Component Analysis: PC1, PC2
# 

# In[28]:


pca = PCA(n_components=2, random_state=42) 
# Only fit to the training set
pca.fit((X_train))
# transform with PCA model from training
principalComponents_train = pca.transform(X_train)
principalComponents_test = pca.transform(X_test)

# Use Pandas DataFrame
X_train = pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
X_train.columns=(data.drop(['id','diagnosis'], axis= 1)).columns
X_test.columns=(data.drop(['id','diagnosis'], axis= 1)).columns
y_train = pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)

X_train['PC1']=principalComponents_train[:,0]
X_train['PC2']=principalComponents_train[:,1]
X_test['PC1']=principalComponents_test[:,0]
X_test['PC2']=principalComponents_test[:,1]


# In[29]:


tr_features=X_train
tr_labels=y_train

val_features = X_test
val_labels=y_test


# # Machine Learning:
# 
# In order to find a good model, several algorithms are tested on the training dataset. A senstivity study using different Hyperparameters of the algorithms are iterated on with GridSearchCV in order optimize each model. The best model is the one that has the highest accuracy without overfitting by looking at both the training data and the validation data results. Computer time does not appear to be an issue for these models, so it has little weight on deciding between models.

# ## GridSearch CV
# 
# class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
# 
# Exhaustive search over specified parameter values for an estimator.
# 
# Important members are fit, predict.
# 
# GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
# 
# The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# In[30]:


print(GridSearchCV)


# ### print results:

# In[31]:


def print_results(results,name,filename_pr):
    with open(filename_pr, mode='w') as file_object:
        print(name,file=file_object)
        print(name)
        print('BEST PARAMS: {}\n'.format(results.best_params_),file=file_object)
        print('BEST PARAMS: {}\n'.format(results.best_params_))
        means = results.cv_results_['mean_test_score']
        stds = results.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, results.cv_results_['params']):
            print('{} {} (+/-{}) for {}'.format(name,round(mean, 3), round(std * 2, 3), params),file=file_object)
            print('{} {} (+/-{}) for {}'.format(name,round(mean, 3), round(std * 2, 3), params))


# # Machine Learning Models:

# ## Support Vector Machine

# ### Hyperparameter used in GridSearchCV
# #### HP1,  kernelstring, optional (default=’rbf’)
# Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
# ###### Details
# A linear kernel type is good when the data is Linearly seperable, which means it can be separated by a single Line.
# A radial basis function (rbf) kernel type is an expontential function of the squared Euclidean distance between two vectors and a constant.  Since the value of RBF kernel decreases with distance and ranges between zero and one, it has a ready interpretation as a similiarity measure.  
# ###### Values chosen
# 'kernel': ['linear','rbf']
# 
# #### HP2,  C:  float, optional (default=1.0)
# Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
# ###### Details
# Regularization is when a penality is applied with increasing value to prevent overfitting.  The inverse of regularization strength means as the value of C goes up, the value of the regularization strength goes down and vice versa.  
# ###### Values chosen
# 'C': [0.1, 1, 10]

# In[32]:


print(SVC())


# In[33]:


SVM_model_dir=os.path.join(currentDirectory,'SVM_model.pkl')
if os.path.exists(SVM_model_dir) == False:
    svc = SVC()
    parameters = {
            'kernel': ['linear','rbf'],
            'C': [0.1, 1, 10]
            }
    cv=GridSearchCV(svc,parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'Support Vector Machine (SVM)','SVM_GridSearchCV_results.txt')
    cv.best_estimator_
    SVM_model_dir=os.path.join(currentDirectory,'SVM_model.pkl')
    joblib.dump(cv.best_estimator_,SVM_model_dir)
else:
    print('Already have SVM')


# ## Random Forest

# ### Hyperparameter used in GridSearchCV
# #### HP1, n_estimators:  integer, optional (default=100)
# The number of trees in the forest.
# 
# Changed in version 0.22: The default value of n_estimators changed from 10 to 100 in 0.22.
# ###### Details
# Usually 500 does the trick and the accuracy and out of bag error doesn't change much after. 
# ###### Values chosen
# 'n_estimators': [500],
# 
# #### HP2, max_depth:  integer or None, optional (default=None)
# The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# ###### Details
# None usually does the trick, but a few shallow trees are tested. 
# ###### Values chosen
# 'max_depth': [5,7,9, None]

# In[34]:


print(RandomForestClassifier())


# In[35]:


RF_model_dir=os.path.join(currentDirectory,'RF_model.pkl')
if os.path.exists(RF_model_dir) == False:
    rf = RandomForestClassifier(oob_score=False)
    parameters = {
            'n_estimators': [500],
            'max_depth': [5,7,9, None]
            }
    cv = GridSearchCV(rf, parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'Random Forest (RF)','RF_GridSearchCV_results.txt')
    cv.best_estimator_
    RF_model_dir=os.path.join(currentDirectory,'RF_model.pkl')
    joblib.dump(cv.best_estimator_,RF_model_dir)
else:
    print('Already have RF')


# ## Gaussian Naive Bayes

# In[36]:


GNB_model_dir=os.path.join(currentDirectory,'GNB_model.pkl')
if os.path.exists(GNB_model_dir) == False:
    gnb = GaussianNB()
    parameters = {'priors':[[0.01, 0.99],[0.1, 0.9], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7],[0.35, 0.65], [0.4, 0.6]]}
    cv=GridSearchCV(gnb,parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'Gussian Naive Bays (GNB)','GNB_GridSearchCV_results.txt')
    cv.best_estimator_
    GNB_model_dir=os.path.join(currentDirectory,'GNB_model.pkl')
    joblib.dump(cv.best_estimator_,GNB_model_dir)
else:
    print('Already have GNB')


# In[37]:


print(GaussianNB)


# ## Decision Tree

# In[38]:


DT_model_dir=os.path.join(currentDirectory,'DT_model.pkl')
if os.path.exists(DT_model_dir) == False:
    dtc = DecisionTreeClassifier()
    parameters = {'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 50], 
              'min_samples_split': [2, 3, 50, 100],
              'min_samples_leaf': [1, 5, 8, 10]
             }
    cv = GridSearchCV(dtc,parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'Decision Tree (DT)','DT_GridSearchCV_results.txt')
    cv.best_estimator_
    DT_model_dir=os.path.join(currentDirectory,'DT_model.pkl')
    joblib.dump(cv.best_estimator_,DT_model_dir)
else:
    print('Already have DT')


# In[39]:


print(DecisionTreeClassifier)


# # Evaluate Models

# In[40]:


models = {} # all models

for mdl in ['SVM', 'RF' , 'GNB','DT']:
    model_path=os.path.join(currentDirectory,'{}_model.pkl')
    models[mdl] = joblib.load(model_path.format(mdl))


# In[41]:


def evaluate_model(name, model, features, labels, y_test_ev, fc):
        start = time()
        pred = model.predict(features)
        end = time()
        y_truth=y_test_ev
        accuracy = round(accuracy_score(labels, pred), 3)
        precision = round(precision_score(labels, pred), 3)
        recall = round(recall_score(labels, pred), 3)
        print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                       accuracy,
                                                                                       precision,
                                                                                       recall,
                                                                                       round((end - start)*1000, 1)))
        
        
        pred=pd.DataFrame(pred)
        pred.columns=['diagnosis']
        # Convert Diagnosis for Cancer from Binary to Categorical
        diagnosis_name={0:'Benign',1:'Malginant'}
        y_truth['diagnosis']=y_truth['diagnosis'].map(diagnosis_name)
        pred['diagnosis']=pred['diagnosis'].map(diagnosis_name)
        class_names = ['Benign','Malginant']        
        cm = confusion_matrix(y_test_ev, pred, class_names)
        
        FP_L='False Positive'
        FP = cm[0][1]
        FN_L='False Negative'
        FN = cm[1][0]
        TP_L='True Positive'
        TP = cm[1][1]
        TN_L='True Negative'
        TN = cm[0][0]

        #TPR_L= 'Sensitivity, hit rate, recall, or true positive rate'
        TPR_L= 'Sensitivity'
        TPR = round(TP/(TP+FN),3)
        #TNR_L= 'Specificity or true negative rate'
        TNR_L= 'Specificity'
        TNR = round(TN/(TN+FP),3) 
        #PPV_L= 'Precision or positive predictive value'
        PPV_L= 'Precision'
        PPV = round(TP/(TP+FP),3)
        #NPV_L= 'Negative predictive value'
        NPV_L= 'NPV'
        NPV = round(TN/(TN+FN),3)
        #FPR_L= 'Fall out or false positive rate'
        FPR_L= 'FPR'
        FPR = round(FP/(FP+TN),3)
        #FNR_L= 'False negative rate'
        FNR_L= 'FNR'
        FNR = round(FN/(TP+FN),3)
        #FDR_L= 'False discovery rate'
        FDR_L= 'FDR'
        FDR = round(FP/(TP+FP),3)

        ACC_L= 'Accuracy'
        ACC = round((TP+TN)/(TP+FP+FN+TN),3)
        
        stats_data = {'Name':name,
                     ACC_L:ACC,
                     FP_L:FP,
                     FN_L:FN,
                     TP_L:TP,
                     TN_L:TN,
                     TPR_L:TPR,
                     TNR_L:TNR,
                     PPV_L:PPV,
                     NPV_L:NPV,
                     FPR_L:FPR,
                     FNR_L:FDR}
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm,cmap=plt.cm.gray_r)
        plt.title('Figure {}.A: {} Confusion Matrix on Test Data'.format(fc,name),y=1.08)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        # Loop over data dimensions and create text annotations.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, cm[i, j],
                               ha="center", va="center", color="r")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('Figure{}.A_{}_Confusion_Matrix.png'.format(fc,name),dpi=400,bbox_inches='tight')
        #plt.show()
        
        if  name == 'RF' or name == 'GB' or name == 'XGB': 
            # Get numerical feature importances
            importances = list(model.feature_importances_)
            importances=100*(importances/max(importances))               
            feature_list = list(features.columns)
            sorted_ID=np.argsort(importances)   
            plt.figure(figsize=[10,10])
            plt.barh(sort_list(feature_list,importances),importances[sorted_ID],align='center')
            plt.title('Figure {}.B: {} Variable Importance Plot'.format(fc,name))
            plt.xlabel('Relative Importance')
            plt.ylabel('Feature') 
            plt.savefig('Figure{}.B_{}_Variable_Importance_Plot.png'.format(fc,name),dpi=300,bbox_inches='tight')
            #plt.show()
        
        return accuracy,name, model, stats_data
        


# In[42]:


def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1)   
    z = [x for _, x in sorted(zipped_pairs)]       
    return z 


# ### Search for best model using test features

# In[43]:


ev_accuracy=[None]*len(models)
ev_name=[None]*len(models)
ev_model=[None]*len(models)
ev_stats=[None]*len(models)
count=1
for name, mdl in models.items():
        y_test_ev=y_test
        ev_accuracy[count-1],ev_name[count-1],ev_model[count-1], ev_stats[count-1] = evaluate_model(name,mdl,val_features, val_labels, y_test_ev,count+1)
        diagnosis_name={'Benign':0,'Malginant':1}
        y_test['diagnosis']=y_test['diagnosis'].map(diagnosis_name)
        count=count+1

    


# In[44]:


best_name=ev_name[ev_accuracy.index(max(ev_accuracy))]    #picks the maximum accuracy
print('Best Model:',best_name,'with Accuracy of ',max(ev_accuracy))   
best_model=ev_model[ev_accuracy.index(max(ev_accuracy))]    #picks the maximum accuracy


# In[45]:


ev_stats=pd.DataFrame(ev_stats)
print(ev_stats.head(10))


# ## Run chosen Algorithm (Random Forest) on wdbc_test.data

# In[46]:


train_data= pd.read_csv('wdbc_train.data', sep=',', header = None , names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",])
test_data= pd.read_csv('wdbc_test.data', sep=',', header = None , names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",])


# In[47]:


test_data.head(5)


# In[48]:


test_data.isnull().sum()


# In[49]:


# Convert Diagnosis for Cancer from Categorical Variable to Binary
diagnosis_num={'B':0,'M':1}
test_data['diagnosis']=test_data['diagnosis'].map(diagnosis_num)


# In[50]:


# Verify Data Changes, look at first 5 rows 
test_data.head(5)


# In[51]:


# Spliting target variable and independent variables
X_train = train_data.drop(['id','diagnosis'], axis= 1)
y_train = train_data.diagnosis


# In[52]:


X_test = test_data.drop(['id','diagnosis'], axis= 1)
y_test = test_data.diagnosis


# In[53]:


#Standardize Data
scaler = StandardScaler()
X_train=StandardScaler().fit_transform(X_train.values)
X_train = pd.DataFrame(X_train)
# X_train.columns=(train_data.drop(['id','diagnosis'], axis= 1)).columns


# In[54]:


scaler = StandardScaler()
X_test=StandardScaler().fit_transform(X_test.values)
X_test = pd.DataFrame(X_test)


# In[55]:


tr_features=X_train
tr_labels=y_train

val_features = X_test
val_labels=y_test


# In[56]:


RF_model_dir=os.path.join(currentDirectory,'RF_model_test.pkl')
if os.path.exists(RF_model_dir) == False:
    rf = RandomForestClassifier(oob_score=False)
    parameters = {
            'n_estimators': [500],
            'max_depth': [5,7,9, None]
            }
    cv = GridSearchCV(rf, parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'Random Forest (RF)','RF_GridSearchCV_test_results.txt')
    cv.best_estimator_
    RF_model_dir=os.path.join(currentDirectory,'RF_model_test.pkl')
    joblib.dump(cv.best_estimator_,RF_model_dir)
else:
    print('Already have RF test')


#  # Conclusions 
#   When it comes to diagnosing breast cancer, we want to make sure we don't have too many false-positives (you don't have cancer, but told you do and go on treatment) or false-negatives (you have cancer, but told you don't and don't get treatment). Therefore, the highest overall accuracy model is chosen.  
# 
#   All of the models performed well after fine tunning their hyperparameters, but the best model is the one the highest overall accuracy.  Out of the 20% of data witheld in this test, only a handful were misdiagnosed.  No model is perfect, but I am happy about how accurate my model is here.  If on average less than a handful of people out of 97 are misdiagnosed, that is a good start for making a model.  Furthermore, the Feature Importance plots show that the "concave points worst" and "concave points mean" were the significant features.  Therefore, the concave point features should be extracted from each future biopsy as a strong predictor for diagnosing breast cancer.   
