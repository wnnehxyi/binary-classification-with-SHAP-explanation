#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import re  
import scipy.stats as st # Calculate the confidence interval
import os 


# ### Load Data 

# In[2]:


datapath = "Your dapatpath"


# In[3]:


# Calculate the confidence interval
files= os.listdir(datapath)

for j in range(len(files)):
    data = pd.read_csv(datapath+files[j])
    print("\n")
    print(files[j])
    
    for i in range(7):
        J = data[data.columns[i]]
        Ans = st.t.interval(alpha=0.95, df=len(J)-1, loc=np.mean(J), scale=st.sem(J))
        print(str(data.columns[i]),": ",Ans)


# In[ ]:


# the descriptive statistics

files= os.listdir(datapath)

for j in range(len(files)):
    data = pd.read_csv(datapath+files[j])
    print(" ",files[j])
    print(data,"\n")

