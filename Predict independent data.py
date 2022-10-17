#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import roc_auc_score

# Modles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier


# ### Load Training Data

# In[ ]:


datapath = "Your dapatpath"
data = pd.read_excel(datapath+"Your_training_data.xlsx")


# In[4]:


# Please select the correct column, this represents the third column.
y = data.iloc[:,2].values  #targets


# In[5]:


# remove the columns you don't want
final_df_1_drop = data.drop(columns=["Date","ID","SMI_cate5"]) 


# In[6]:


scaler_z = StandardScaler() # create scaler method 
z_data = scaler_z.fit_transform(final_df_1_drop) # fit and transform the data
pd_z_data = pd.DataFrame(z_data, columns = final_df_1_drop.columns) #convert to dataframe


# ### Load Independent Data

# In[ ]:


datapath = "Your dapatpath"
indata = pd.read_excel(datapath+"Your_test_data.xlsx")


# In[9]:


# Please select the correct column, this represents the third column.
in_y = indata.iloc[:,2].values # targets


# In[10]:


in_final_df_drop = indata.drop(columns=["Date","ID","SMI_cate5"])


# In[11]:


in_z_data = scaler_z.fit_transform(in_final_df_drop)
in_pd_z_data = pd.DataFrame(in_z_data, columns = in_final_df_drop.columns)


# ### Load Models 

# In[12]:


classifier_LR = LogisticRegression()
classifier_RF = RandomForestClassifier(random_state=1)
classifier_SVM = SVC(probability=True)
classifier_XGB = XGBClassifier()


# ### All training data to fit a model and Predict the independent data (Using LR as an example)

# In[13]:


all_X_train = pd_z_data
all_Y_train = y
X_124_test = in_pd_z_data
Y_124_test = in_y


# In[14]:


# Use all training data(493) to create a model
# Use this model to predict independent data(124)
The_model_LR = classifier_LR.fit(all_X_train, all_Y_train)
print('label = 1:',sum(all_Y_train))
print('label = 0:',len(all_Y_train) - sum(all_Y_train))

in_predicted_LR = The_model_LR.predict(X_124_test)
print(metrics.classification_report(Y_124_test, in_predicted_LR))


# In[17]:


in_Pre_LR = metrics.precision_score(Y_124_test, in_predicted_LR)
print('Precision_predict:',in_Pre_LR)
in_Sen_LR = metrics.recall_score(Y_124_test, in_predicted_LR)
print('Sensitivity_predict:',in_Sen_LR)
in_Spe_LR = metrics.recall_score(Y_124_test, in_predicted_LR, pos_label=0)
print('Specificity_predict:',in_Spe_LR)
in_F1_LR = metrics.f1_score(Y_124_test, in_predicted_LR)
print('F1_predict:',in_F1_LR)
in_Acc_LR = metrics.accuracy_score(Y_124_test, in_predicted_LR)
print('accuracy_predict:',in_Acc_LR)
yPred_p_LR = The_model_LR.predict_proba(X_124_test)[:,1]
print('auc_predict:',roc_auc_score(Y_124_test, yPred_p_LR),"\n")

