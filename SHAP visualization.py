#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# All training data to fit a model and create an explainer
# Visualize independent data
# Algorithm: Random Forest


# In[17]:


import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set_theme(color_codes=True)

# Modles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

# SHAP
import shap
shap.initjs()


# ### Load Training Data

# In[ ]:


datapath = "Your dapatpath"
data = pd.read_excel(datapath+"Your_training_data.xlsx")


# In[3]:


# Please select the correct column, this represents the third column.
y = data.iloc[:,2].values  #targets


# In[4]:


# remove the columns you don't want
final_df_1_drop = data.drop(columns=["Date","ID","SMI_cate5"]) 


# In[5]:


scaler_z = StandardScaler() # create scaler method 
z_data = scaler_z.fit_transform(final_df_1_drop) # fit and transform the data
pd_z_data = pd.DataFrame(z_data, columns = final_df_1_drop.columns) #convert to dataframe


# ### Load Independent Data

# In[ ]:


datapath = "Your dapatpath"
indata = pd.read_excel(datapath+"Your_test_data.xlsx")


# In[7]:


# Please select the correct column, this represents the third column.
in_y = indata.iloc[:,2].values # targets


# In[8]:


in_final_df_drop = indata.drop(columns=["Date","ID","SMI_cate5"])


# In[9]:


in_z_data = scaler_z.fit_transform(in_final_df_drop)
in_pd_z_data = pd.DataFrame(in_z_data, columns = in_final_df_drop.columns)


# ### Load Models 

# In[10]:


classifier_LR = LogisticRegression()
classifier_RF = RandomForestClassifier(random_state=1)
classifier_SVM = SVC(probability=True)
classifier_XGB = XGBClassifier()


# ### Create explainer

# In[11]:


all_X_train = pd_z_data
all_Y_train = y
X_124_test = in_pd_z_data
Y_124_test = in_y


# In[12]:


The_model_RF = classifier_RF.fit(all_X_train, all_Y_train)
explainer_RF = shap.TreeExplainer(The_model_RF)
shap_values_RF_inde_test = explainer_RF.shap_values(X_124_test)


# ## SHAP Summary Plot

# In[15]:


shap.summary_plot(shap_values_RF_inde_test[1], X_124_test)


# ## SHAP Dependence Plot

# In[18]:


RF_five = ["Alb_change", "Ascites_1", "ECOG", "NLR_change", "rBMI_change", "Age", "PLR_change"]

for i in range(7):
    _,ax = plt.subplots(figsize=(7.5,5.5))
    plt.xlim((-6, 6))
    plt.ylim((-0.25, 0.55))
    plt.axhline(y=0, color = "red", linestyle = "-") 
    plt.grid(False)
    
    shap.dependence_plot(RF_five[i], shap_values_RF_inde_test[1], X_124_test, interaction_index=None, ax=ax, show=False)
    
    f = plt.gcf() # gcf: Get current Figure
    pic_save_datapath = "Your location"
    f.savefig(pic_save_datapath+'SHAP dependence plot_RF_'+RF_five[i]+'.png',dpi = 96,bbox_inches="tight")
    plt.show()
    f.clear()  # clear memory


# In[19]:


# https://shap-lrjball.readthedocs.io/en/latest/generated/shap.dependence_plot.html
# shap.approximate_interactions

cli_feature = ["Age","Ascites_1","ECOG"]

for j in range(len(cli_feature)):
    inds = shap.approximate_interactions(cli_feature[j], shap_values_RF_inde_test[1], X_124_test)

    # make plots colored by each of the three possible interacting features
    for i in range(len(inds)):
        shap.dependence_plot(cli_feature[j], shap_values_RF_inde_test[1],  X_124_test, interaction_index=inds[i],show = False)
            
        f = plt.gcf() ## gcf: Get current Figure
        pic_save_datapath = "Your location"
        f.savefig(pic_save_datapath+"SHAP approximate_interactions_RF_"+cli_feature[j]+"_"+str(inds[i])+".png",dpi = 96,bbox_inches="tight")
        plt.show()
        f.clear() # clear memory


# ## SHAP Force Plot 

# In[20]:


The_model_RF = classifier_RF.fit(all_X_train, all_Y_train)
explainer_RF = shap.TreeExplainer(The_model_RF)
shap_values_RF_inde_test = explainer_RF.shap_values(X_124_test)


# In[22]:


# round to three decimal places (personal preference)
in_pd_z_data_round3 = in_pd_z_data.copy()
in_pd_z_data_round3 = in_pd_z_data_round3.round(3)


# In[23]:


def shap_plot_save(j):
    explainerModel = shap.TreeExplainer(The_model_RF)
    shap_values_Model = explainerModel.shap_values(in_pd_z_data_round3)
    p = shap.force_plot(explainerModel.expected_value[1], shap_values_Model[1][j], in_pd_z_data_round3.iloc[[j]], matplotlib = True,show=False)
    
    plt.grid(False)
    pic_save_datapath = "Your location"
    plt.savefig(pic_save_datapath + str(indata["ID"][j]) +'_RF.png',bbox_inches="tight",facecolor='white')
    plt.close()
    
    return(p)


# In[24]:


for i in range(X):  #X:Number of pictures
    shap_plot_save(i)

