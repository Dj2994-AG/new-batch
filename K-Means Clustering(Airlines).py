#!/usr/bin/env python
# coding: utf-8

# In[39]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# In[41]:


airline=pd.read_csv('C:/Users/A/Downloads/EastWestAirlines.csv')
airline


# In[42]:


airline2=airline.drop(['ID#'],axis=1)
airline2


# In[43]:


airline2.info()


# In[44]:


# Normalize heterogenous numerical data
airline2_norm=pd.DataFrame(normalize(airline2),columns=airline2.columns)
airline2_norm


# In[45]:


# within-cluster sum-of-squares criterion 
wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(airline2_norm)
    wcss.append(kmeans.inertia_)


# In[46]:


#Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,11),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[47]:


# Cluster algorithm using K=4
clusters4=KMeans(4,random_state=30).fit(airline2_norm)
clusters4


# In[48]:



clusters4.labels_


# In[49]:


# Assign clusters to the data set
airline4=airline2.copy()
airline4['clusters4id']=clusters4.labels_
airline4


# In[50]:


# Compute the centroids for K=4 clusters with 11 variables
clusters4.cluster_centers_


# In[51]:


# Group data by Clusters (K=4)
airline4.groupby('clusters4id').agg(['mean']).reset_index()


# In[52]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(airline4['clusters4id'],airline4['Balance'], c=clusters4.labels_)


# In[53]:


# Cluster algorithm using K=5
clusters5=KMeans(5,random_state=30).fit(airline2_norm)
clusters5


# In[54]:


clusters5.labels_


# In[55]:


# Assign clusters to the data set
airline5=airline2.copy()
airline5['clusters5id']=clusters5.labels_
airline5


# In[56]:



# Compute the centroids for K=5 clusters with 11 variables
clusters5.cluster_centers_


# In[57]:


# Group data by Clusters (K=5)
airline5.groupby('clusters5id').agg(['mean']).reset_index()


# In[58]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(airline5['clusters5id'],airline5['Balance'], c=clusters5.labels_)


# In[ ]:




