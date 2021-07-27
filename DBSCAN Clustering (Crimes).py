#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[2]:


# Import Dataset
crime=pd.read_csv('C:/Users/A/Downloads/crime_data.csv')
crime


# In[3]:


crime.info()


# In[4]:


crime.drop(['Unnamed: 0'],axis=1,inplace=True)
crime


# In[5]:


# Normalize heterogenous numerical data using standard scalar fit transform to dataset
crime_norm=StandardScaler().fit_transform(crime)
crime_norm


# In[6]:


# DBSCAN Clustering
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(crime_norm)


# In[7]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[8]:


# Adding clusters to dataset
crime['clusters']=dbscan.labels_
crime


# In[9]:


crime.groupby('clusters').agg(['mean']).reset_index()


# In[10]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(crime['clusters'],crime['UrbanPop'], c=dbscan.labels_)


# In[ ]:




