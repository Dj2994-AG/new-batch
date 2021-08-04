#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[12]:


movie=pd.read_csv('C:/Users/A/Downloads/my_movies_final.csv')
movie


# In[14]:


# With 10% Support
frequent_itemsets=apriori(movie,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[16]:


# with 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules


# In[17]:


## A leverage value of 0 indicates independence. Range will be [-1 1]
## A high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]


# In[18]:


rules.sort_values('lift',ascending=False)


# In[19]:


# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]


# In[20]:


# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[22]:


# With 20% Support
frequent_itemsets2=apriori(movie,min_support=0.20,use_colnames=True)
frequent_itemsets2


# In[23]:


# With 60% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2


# In[24]:


# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[26]:


# With 5% Support
frequent_itemsets3=apriori(movie,min_support=0.05,use_colnames=True)
frequent_itemsets3


# In[27]:


# With 80% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3


# In[28]:


rules3[rules3.lift>1]


# In[29]:


# visualization of obtained rule
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[ ]:




