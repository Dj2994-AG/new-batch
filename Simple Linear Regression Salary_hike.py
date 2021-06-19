#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[5]:


dataset=pd.read_csv('C:/Users/A/Downloads/Salary_Data.csv')
dataset


# In[6]:


dataset.info()


# In[11]:


sns.distplot(dataset['YearsExperience'])


# In[12]:


sns.distplot(dataset['Salary'])


# In[13]:


dataset.corr()


# In[14]:


sns.regplot(x=dataset['YearsExperience'],y=dataset['Salary'])


# In[15]:


model=smf.ols("Salary~YearsExperience",data=dataset).fit()


# In[16]:


model.params


# In[17]:


model.tvalues, model.pvalues


# In[18]:


model.rsquared , model.rsquared_adj


# In[19]:


Salary = (25792.200199) + (9449.962321)*(3)
Salary


# In[20]:


new_data=pd.Series([3,5])
new_data


# In[21]:


data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])
data_pred


# In[22]:


model.predict(data_pred)


# In[ ]:




