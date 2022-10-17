#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set_theme(color_codes=True)

# Modles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier


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


# ### ROC curve (Using LR as an example)

# In[9]:


tprs = []
aucs = []
base_fpr = np.linspace(0, 1, 501)

for i in range(500):
    X_train, X_test, Y_train, Y_test = train_test_split(pd_z_data, y, test_size = 0.3, random_state=i)
    classifier_LR.fit(X_train, Y_train)
    
    predicted_LR_p =classifier_LR.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(Y_test, predicted_LR_p)
    
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)
    
tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

mean_auc = auc(base_fpr, mean_tprs)
std_auc = np.std(aucs)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std


plt.figure(dpi = 100)
plt.plot(base_fpr, mean_tprs, 'royalblue', alpha = 0.8, label='Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'cornflowerblue', alpha = 0.2)
plt.plot([0, 1], [0, 1], linestyle = '--', lw = 1.5, color = 'r')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.legend(loc="lower right")
plt.title('LogisticRegression (ROC) curve')
plt.grid(False)
plt.show()

