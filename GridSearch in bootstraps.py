#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np 
import re  
import scipy.stats as st 
from sklearn.preprocessing import StandardScaler  # Z-transformation
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set_theme(color_codes=True)

from sklearn.model_selection import GridSearchCV  # grid search methods
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import roc_curve, auc

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier

# Balanced classes
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# ### Load Data 

# In[2]:


datapath = "Your dapatpath"
data = pd.read_excel(datapath+"Your_training_data.xlsx")


# In[ ]:


# the third column is the target.
y = data.iloc[:,2].values


# In[ ]:


# remove the columns (e.g. target)
final_df_1_drop = data.drop(columns=["Date","ID","SMI_cate5"]) 


# In[ ]:


scaler_z = StandardScaler()  # create scaler method 
z_data = scaler_z.fit_transform(final_df_1_drop)  # fit and transform the data
pd_z_data = pd.DataFrame(z_data, columns = final_df_1_drop.columns)  #convert to dataframe


# ### GridSearchCV 

# In[12]:


# model
classifiers = [
    LogisticRegression()]


# In[13]:


# Enter the combinations of parameter

logregress_parameters = {
    'classifier__C':[0.01, 0.05, 0.1, 0.25, 0.5, 1, 5, 10, 15, 20]
    #'classifier__C': np.logspace(-3,3,7)*2  #[2.e-03 2.e-02 2.e-01 2.e+00 2.e+01 2.e+02 2.e+03]
}


# In[14]:


parameters = [
    logregress_parameters]


# In[27]:


best_para_Ans = []
best_scor_Ans = []
best_mean_Ans = []
best_std_Ans = []


for j in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(pd_z_data, y, test_size = 0.3, random_state=j)
    
    ########################################################
    ## Oversampling method to balance 
    over = RandomOverSampler(sampling_strategy='auto', random_state = 1)
    all_X_smote_F, all_y_smote_F = over.fit_resample(X_train, Y_train)    
    ########################################################
    
    for i, classifier in enumerate(classifiers):
    # create a Pipeline object
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])

        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  scoring= 'roc_auc',     # metric for scoring (Use sorted(sklearn.metrics.SCORERS.keys()) to get valid options.)
                  cv=10)                 # number of folds
        
        ########################################################
        #The number of labels after balance
        clf.fit(all_X_smote_F, all_y_smote_F)
        #clf.fit(X_train, Y_train)
        
        print('label = 1:',sum(all_y_smote_F))
        print('label = 0:',len(all_y_smote_F) - sum(all_y_smote_F))
        print('before balance label = 1:',sum(Y_train))
        print('before balance label = 0:',len(Y_train) - sum(Y_train),"\n")
        ########################################################

        print("Random_state :", j)
        print("Tuned Hyperparameters :", clf.best_params_)
        print("Recall :", clf.best_score_)
        
        best_para = clf.best_params_
        best_para_Ans.append(best_para)
        best_scor = clf.best_score_
        best_scor_Ans.append(best_scor)
        
        
        # mean and standard deviation of each cross-validation
        print("10-fold mean :", clf.cv_results_['mean_test_score'])
        print("10-fold std :", clf.cv_results_['std_test_score'],"\n")
        
        best_mean = clf.cv_results_['mean_test_score']
        best_mean_Ans.append(best_mean)
        
        best_std = clf.cv_results_['std_test_score']
        best_std_Ans.append(best_std)


# In[28]:


# The best parameter settings for each bootstrap
key_par_ans = []
value_par_ans = []

for i in range(len(best_para_Ans)):
    for key, value in best_para_Ans[i].items():
            if key.startswith('classifier__C'):
                    print(key, value)

                    key_par = key
                    key_par_ans.append(key_par)
                    
                    print(value,"\n")
                    value_par = value
                    value_par_ans.append(value_par)


# In[29]:


# Save each result
save_datapath = "Your location"

df_grid = pd.DataFrame()
df_grid['C'] = key_par_ans
df_grid['C_Ans'] = value_par_ans
df_grid['best_AUC'] = best_scor_Ans
df_grid.to_csv(save_datapath+"__rename__.csv")


# In[29]:


# Use list.count() to count the most frequent parameter in bootstraps
find_par_ans = []

for i in range(len(best_para_Ans)):
    for key, value in best_para_Ans[i].items():
            if key.startswith('classifier__C'):
                    print(value)
                    
                    find_par = value
                    find_par_ans.append(find_par)


# In[30]:


print('LR = 0.01: ', find_par_ans.count(0.01))
print('LR = 0.05: ', find_par_ans.count(0.05))
print('LR = 0.1: ', find_par_ans.count(0.1))
print('LR = 0.25: ', find_par_ans.count(0.25))
print('LR = 0.5: ', find_par_ans.count(0.5))
print('LR = 1: ', find_par_ans.count(1))
print('LR = 5: ', find_par_ans.count(5))
print('LR = 10: ', find_par_ans.count(10))
print('LR = 15: ', find_par_ans.count(15))
print('LR = 20: ', find_par_ans.count(20),"\n")


# In[31]:


# mean of each cross-validation in bootstraps
mean_LR_Ans = []

for i in range(5):
    data_mean_LR = np.array(best_mean_Ans[i])
    mean_LR = data_mean_LR.tolist()
    mean_LR_Ans.append(mean_LR)


# In[32]:


# Store as DataFrame (rows: number of bootstraps, columns: parameters)
pd_mean_LR = pd.DataFrame(mean_LR_Ans, columns = ['0.01','0.05','0.1','0.25','0.5','1','5','10','15','20'])
pd_mean_LR


# In[33]:


# descriptive statistics
pd_mean_LR_DES = pd_mean_LR.describe()
pd_mean_LR_DES


# In[30]:


save_datapath = "Your location"

pd_mean_LR.to_csv(save_datapath+"__rename__.csv")
pd_mean_LR_DES.to_csv(save_datapath+"__rename__.csv")


# In[34]:


# standard deviation of each cross-validation in bootstraps
std_LR_Ans = []

for i in range(5): 
    data_std_LR = np.array(best_std_Ans[i])
    std_LR = data_std_LR.tolist()
    std_LR_Ans.append(std_LR)


# In[35]:


pd_std_LR = pd.DataFrame(std_LR_Ans, columns = ['0.01','0.05','0.1','0.25','0.5','1','5','10','15','20'])
pd_std_LR


# In[36]:


pd_std_LR_DES = pd_std_LR.describe()
pd_std_LR_DES


# In[34]:


save_datapath = "Your location"

pd_std_LR.to_csv(save_datapath+"__rename__.csv")
pd_std_LR_DES.to_csv(save_datapath+"__Rename__.csv")


# In[37]:


# the best score for metrics in bootstraps

acc_10f_LR_Ans = []

for i in range(5): 
    data_acc_10f_LR = np.array(best_scor_Ans[i])
    acc_10f_LR = data_acc_10f_LR.tolist()
    acc_10f_LR_Ans.append(acc_10f_LR)


# In[38]:


pd_acc_10f_LR = pd.DataFrame(acc_10f_LR_Ans, columns = ['AUC_10fold'])
pd_acc_10f_LR


# In[39]:


pd_acc_10f_LR_DES = pd_acc_10f_LR.describe()
pd_acc_10f_LR_DES


# In[38]:


save_datapath = "Your location"

pd_acc_10f_LR.to_csv(save_datapath+"__rename__.csv")
pd_acc_10f_LR_DES.to_csv(save_datapath+"__Rename__.csv")

