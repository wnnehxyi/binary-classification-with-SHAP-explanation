#!/usr/bin/env python
# coding: utf-8

# In[8]:


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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier 

# Balanced classes
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# ### Load Data

# In[ ]:


datapath = "Your dapatpath"
data = pd.read_excel(datapath+"Your_training_data.xlsx")


# In[3]:


# the third column is the target.
y = data.iloc[:,2].values


# In[4]:


# remove the columns (e.g. target)
final_df_1_drop = data.drop(columns=["Date","ID","SMI_cate5"]) 


# In[5]:


scaler_z = StandardScaler() # create scaler method 
z_data = scaler_z.fit_transform(final_df_1_drop) # fit and transform the data
pd_z_data = pd.DataFrame(z_data, columns = final_df_1_drop.columns) #convert to dataframe


# ### Load Models 

# In[6]:


classifier_LR = LogisticRegression()
classifier_RF = RandomForestClassifier(random_state=1)
classifier_KNN = KNeighborsClassifier()
classifier_SVM = SVC(probability=True)
classifier_XGB = XGBClassifier()
classifier_LGB = LGBMClassifier()
classifier_NB = BernoulliNB()

model1_1 = LogisticRegression()
model1_2 = SVC(probability=True, random_state=0)
classifier_Ensemble = VotingClassifier(estimators = [('logistic',model1_1),('svm',model1_2)], voting = 'soft')


# ### 500 bootstraps (Using LR as an example)

# In[9]:


#scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
#for j in range(len(scales)):
    
    precision_ans_ov_LR = []
    sensitivity_ans_ov_LR = []
    specificity_ans_ov_LR = []
    F1_ans_ov_LR = []
    accuracy_ans_ov_LR = []
    auc_ans_ov_LR = []

    # X_train: Training data、X_test: Validation data
    # Randomly split the data and made 500 predictions
    for i in range(500):
        print("Random_state :", i)
        X_train, X_test, Y_train, Y_test = train_test_split(pd_z_data, y, test_size = 0.3, random_state=i)

        """
        ## Oversampling or Undersampling can be used separately.
        ## Here, both types of methods are used together.

        ### 1. Oversample the minority class as much as the majority class.
        over = RandomOverSampler(sampling_strategy='auto', random_state = 1)
        all_X_smote_F, all_y_smote_F = over.fit_resample(X_train, Y_train)

        ### 2. Try two types of data at different scales. (Use nested loop)    
        class1 = sum(Y_train)*scales[j]
        class0 = int(sum(Y_train)*scales[j])
        print("Persentage: ",scales[j])
        #print(class1)
        #print(class0)
        sampling_strategy = {0: class0, 1: sum(all_y_smote_F)}

        under = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state = 1)
        X_smote, y_smote = under.fit_resample(all_X_smote_F, all_y_smote_F)
        """

        # If it is balanced, need to change the variable: X_train→ X_smote, Y_train→y_smote
        classifier_LR.fit(X_train, Y_train)
        print('label = 1:',sum(Y_train))
        print('label = 0:',len(Y_train) - sum(Y_train))
        print('before balance label = 1:',sum(Y_train))
        print('before balance label = 0:',len(Y_train) - sum(Y_train))

        # Predict
        predicted_LR = classifier_LR.predict(X_test)
        print(metrics.classification_report(Y_test, predicted_LR))

        Pre_LR = metrics.precision_score(Y_test, predicted_LR)
        print('Precision_predict:',Pre_LR)
        Sen_LR = metrics.recall_score(Y_test, predicted_LR)
        print('Sensitivity_predict:',Sen_LR)
        Spe_LR = metrics.recall_score(Y_test, predicted_LR, pos_label=0)
        print('Specificity_predict:',Spe_LR)
        F1_LR = metrics.f1_score(Y_test, predicted_LR)
        print('F1_predict:',F1_LR)
        Acc_LR = metrics.accuracy_score(Y_test, predicted_LR)
        print('accuracy_predict:',Acc_LR)
        predicted_LR_p = classifier_LR.predict_proba(X_test)[:,1]
        auc_LR = roc_auc_score(Y_test, predicted_LR_p)
        print('auc_predict:',roc_auc_score(Y_test, predicted_LR_p),"\n")

        # Save each result
        precision_ans_ov_LR.append(Pre_LR)
        sensitivity_ans_ov_LR.append(Sen_LR)
        specificity_ans_ov_LR.append(Spe_LR)
        F1_ans_ov_LR.append(F1_LR)
        accuracy_ans_ov_LR.append(Acc_LR)
        auc_ans_ov_LR.append(auc_LR)    

        save_datapath = "Your location"
        df_me_ov_LR = pd.DataFrame()
        df_me_ov_LR['precision'] = precision_ans_ov_LR
        df_me_ov_LR['sensitivity'] = sensitivity_ans_ov_LR
        df_me_ov_LR['specificity'] = specificity_ans_ov_LR
        df_me_ov_LR['F1'] = F1_ans_ov_LR
        df_me_ov_LR['accuracy'] = accuracy_ans_ov_LR
        df_me_ov_LR['AUC'] = auc_ans_ov_LR
        df_me_ov_LR.to_csv(save_datapath+"__rename__.csv")

    # Save total result
    DES_ov_LR = df_me_ov_LR.describe()
    DES_ov_LR.to_csv(save_datapath+"__Rename__.csv")
    print(DES_ov_LR, "\n")

