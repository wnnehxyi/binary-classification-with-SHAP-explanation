#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

# Modles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

# SHAP
import shap
shap.initjs()


# ### Load Data

# In[ ]:


datapath = "Your dapatpath"
data = pd.read_excel(datapath+"Your_training_data.xlsx")


# In[3]:


# Please select the correct column, this represents the third column.
y = data.iloc[:,2].values  #targets


# In[4]:


final_df_1_drop = data.drop(columns=["Date","ID","SMI_cate5"]) # remove the columns you don't want


# In[5]:


scaler_z = StandardScaler() # create scaler method 
z_data = scaler_z.fit_transform(final_df_1_drop) # fit and transform the data
pd_z_data = pd.DataFrame(z_data, columns = final_df_1_drop.columns) #convert to dataframe


# ### Load Models 

# In[6]:


classifier_LR = LogisticRegression()
classifier_RF = RandomForestClassifier(random_state=1)
classifier_SVM = SVC(probability=True)
classifier_XGB = XGBClassifier()


# ### SHAP 500 bootstraps (Using RF as an example)

# ####  Please choose the best explainer for the model. 
# (https://shap-lrjball.readthedocs.io/en/latest/generated/shap.Explainer.html)
# 
# → Core Explainers → shap.Expaliner, shap.TreeExplainer,... etc.

# In[7]:


First_Ans_RF_trex_500 = []
Second_Ans_RF_trex_500 = []
Third_Ans_RF_trex_500 = []
Four_Ans_RF_trex_500 = []
Five_Ans_RF_trex_500 = []

# X_train: Training data、X_test: Validation data
for i in range(500):
    print("Random_state :", i)
    X_train, X_test, Y_train, Y_test = train_test_split(pd_z_data, y, test_size = 0.3, random_state=i)

    The_model_RF = classifier_RF.fit(X_train, Y_train)
    explainer_RF = shap.TreeExplainer(The_model_RF)
    shap_values_RF = explainer_RF.shap_values(X_test)
    
    shap_sum = np.abs(shap_values_RF[1]).mean(axis=0)
    
    importance_df = pd.DataFrame([X_test.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    print(importance_df)
    
    # Save each result
    save_500_datapath = "Your location"
    importance_df.to_csv(save_500_datapath+str(i)+"_RF_500_SHAP_Rank.csv")
    
    
    numpy_array = importance_df.to_numpy().T
    print("First: ",numpy_array[0][0])
    First = numpy_array[0][0]
    First_Ans_RF_trex_500.append(First)
    
    print("Second: ",numpy_array[0][1])
    Second = numpy_array[0][1]
    Second_Ans_RF_trex_500.append(Second)
    
    print("Third: ",numpy_array[0][2])
    Third = numpy_array[0][2]
    Third_Ans_RF_trex_500.append(Third)
    
    print("Four: ",numpy_array[0][3])
    Four = numpy_array[0][3]
    Four_Ans_RF_trex_500.append(Four)
    
    print("Five: ",numpy_array[0][4],"\n")
    Five = numpy_array[0][4]
    Five_Ans_RF_trex_500.append(Five)


# In[9]:


save_datapath = "Your location"
df_RF_trex_500 = pd.DataFrame()
df_RF_trex_500['First'] = First_Ans_RF_trex_500
df_RF_trex_500['Second'] = Second_Ans_RF_trex_500
df_RF_trex_500['Third'] = Third_Ans_RF_trex_500
df_RF_trex_500['Four'] = Four_Ans_RF_trex_500
df_RF_trex_500['Five'] = Five_Ans_RF_trex_500

# Save total result
df_RF_trex_500.to_csv(save_datapath+"RF_500_SHAP__Rank.csv")


# In[10]:


# Count the number of features appearing for each ranking
print(df_RF_trex_500[str(df_RF_trex_500.columns[0])].value_counts(sort = True),"\n")
print(df_RF_trex_500[str(df_RF_trex_500.columns[1])].value_counts(sort = True),"\n")
print(df_RF_trex_500[str(df_RF_trex_500.columns[2])].value_counts(sort = True),"\n")
print(df_RF_trex_500[str(df_RF_trex_500.columns[3])].value_counts(sort = True),"\n")
print(df_RF_trex_500[str(df_RF_trex_500.columns[4])].value_counts(sort = True),"\n")

