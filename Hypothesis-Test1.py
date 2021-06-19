#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import the libraries
import pandas as pd
import scipy 
import numpy as np
from scipy import stats


# In[6]:


data=pd.read_csv('C:/Users/A/Downloads/Cutlets.csv')


# In[7]:


data


# In[10]:


data.head(5)


# In[11]:


unitA=pd.Series(data.iloc[:,0])


# In[12]:


unitA


# In[13]:


unitB=pd.Series(data.iloc[:,1])


# In[14]:


unitB


# In[28]:


p_value=stats.ttest_ind(unitA,unitB)


# In[29]:


p_value


# In[30]:


p_value[1]     # 2-tail probability


# In[50]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


# In[33]:


data=pd.read_csv('C:/Users/A/Downloads/Costomer+OrderForm.csv')


# In[42]:


data


# In[51]:


data.Phillippines.value_counts()


# In[52]:


data.Indonesia.value_counts()


# In[53]:


data.Malta.value_counts()


# In[54]:


data.India.value_counts()


# In[55]:


obs=np.array([[271,267,269,280],[29,33,31,20]])


# In[48]:


obs


# In[49]:


chi2_contingency(obs)


# In[56]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm


# In[57]:


data=pd.read_csv('C:/Users/A/Downloads/LabTAT.csv')


# In[58]:


data


# In[59]:


p_value=stats.f_oneway(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],data.iloc[:,3])


# In[60]:


p_value


# In[61]:



p_value[1]


# In[62]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


# In[63]:


data=pd.read_csv('C:/Users/A/Downloads/BuyerRatio.csv')


# In[64]:


data.head()


# In[65]:


data


# In[66]:


obs=np.array([[50,142,131,70],[435,1523,1356,750]])


# In[67]:


obs


# In[68]:


chi2_contingency(obs) # o/p is (Chi2 stats value, p_value, df, expected obsvations)


# In[ ]:




