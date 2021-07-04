#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.api as smf
import numpy as np


# In[9]:


#importing data 
compdata = pd.read_csv('C:/Users/A/Downloads/50_Startups.csv')
compdata.head()


# In[10]:


# checking data details and checking for null data.
compdata.info()


# In[15]:


#compareing compete data by grouping by state.
compdata.groupby("State").mean()


# In[16]:


compdata.head()


# In[17]:


sns.set_style(style='darkgrid')
sns.pairplot(compdata)


# In[19]:


# As state is a categorical variable, we are getting dummies to get the values of unique states.
compdata = pd.get_dummies(compdata,drop_first= True)


# In[20]:


compdata.head()


# In[21]:


# Separating the dataset into x and y 
x = compdata.drop("Profit", axis= 1)
y = compdata["Profit"]


# In[22]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 4)


# In[23]:


m1 = smf.OLS(y_train, x_train).fit()


# In[24]:


m1.summary2()


# In[25]:


y_train2 = np.sqrt(y_train)


# In[26]:


y_train2.head()


# In[27]:


m2 = smf.OLS(y_train2,x_train).fit()


# In[28]:


m2.summary2()


# In[29]:


y_train3 = np.log(y_train)


# In[30]:


y_train3.head()


# In[31]:


m3 = smf.OLS(y_train3,x_train).fit()


# In[32]:


m3.summary2()


# In[33]:


x_train1 = x_train.drop(["State_Florida","State_New York"],axis = 1)


# In[34]:


x_train1.head()


# In[35]:


m4 = smf.OLS(y_train3,x_train1).fit()


# In[37]:


m4.summary2()


# In[38]:


x_train2 = x_train.drop(["State_Florida","State_New York"],axis = 1)


# In[39]:


x_train2.head()


# In[40]:


m5 = smf.OLS(y_train,x_train1).fit()


# In[41]:


m5.summary2()


# In[42]:


m5.rsquared


# In[43]:


data = [{'R Square Values': m1.rsquared, 'AIC':m1.aic}, {'R Square Values': m2.rsquared, 'AIC':m2.aic},{'R Square Values': m3.rsquared, 'AIC':m3.aic}, {'R Square Values': m4.rsquared, 'AIC':m4.aic}, {'R Square Values': m5.rsquared, 'AIC':m5.aic}]

# Lists of dictionaries and row index.
df = pd.DataFrame(data, index =['Model without any modification', 'Y transformed using sqrt','Y transformed using natural log','Y -ln with insignificant variables removed','Insignificant values removed with original model'])
  
# Print the data
df


# In[46]:


#we can see that the model with insignificant state columns removed and transformed using natural log yeilds the best AIC scores, so we can say that this is the best model.


# In[ ]:




