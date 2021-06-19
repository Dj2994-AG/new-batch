#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[3]:


dataset=pd.read_csv('C:/Users/A/Downloads/delivery_time.csv')


# In[4]:


dataset


# In[5]:


dataset.info()


# In[6]:


sns.distplot(dataset['Delivery Time'])


# In[7]:


sns.distplot(dataset['Sorting Time'])


# In[8]:


dataset=dataset.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
dataset


# In[9]:


dataset.corr()


# In[13]:


sns.regplot(x=dataset['sorting_time'],y=dataset['delivery_time'])


# In[14]:


model=smf.ols("delivery_time~sorting_time",data=dataset).fit()


# In[15]:


# Finding Coefficient parameters
model.params


# In[16]:


# Finding tvalues and pvalues
model.tvalues , model.pvalues


# In[17]:


# Finding Rsquared Values
model.rsquared , model.rsquared_adj


# In[18]:


# Manual prediction for say sorting time 5
delivery_time = (6.582734) + (1.649020)*(5)
delivery_time


# In[19]:


# Automatic Prediction for say sorting time 5, 8
new_data=pd.Series([5,8])
new_data


# In[20]:


data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred


# In[21]:


model.predict(data_pred)


# In[ ]:




