#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


df1 = pd.read_csv("C:/Users/A/Downloads/Q7.csv")


# In[20]:


df1.columns


# In[22]:


pd.crosstab(df1.Points,df1.Weigh).plot(kind="bar")


# In[23]:


df1["Points"].value_counts()
df1.Points.value_counts().plot(kind="pie")


# In[24]:


plt.hist(df1['Score']) 


# In[25]:


plt.hist(df1['Weigh']) 


# In[26]:


plt.hist(df1['Points']) 


# In[12]:


x = [-3, 5, 7]
y = [10, 2, 5]


# In[13]:


fig = plt.figure(figsize=(15,3))

plt.plot(x, y)
plt.xlim(0, 10)
plt.ylim(-3, 8)
plt.xlabel('X Axis')
plt.ylabel('Y axis')


# In[14]:


fig, ax = plt.subplots(nrows=1, ncols=1)


# In[15]:


fig, axs = plt.subplots(2, 4)


# In[27]:


plt.boxplot(df1['Points'],vert = True)


# In[49]:


import seaborn as sns

plt.boxplot(df1['Points'],vert = True)
plt.boxplot(df1['Score'],vert = True)
plt.boxplot(df1['Weigh'],vert = True)


# In[29]:


plt.boxplot(df1['Weigh'],vert = True)


# In[30]:


import seaborn as sns


# In[35]:


df1.head()


# In[53]:


sns.boxplot(y='Points', data=df1)


# In[54]:


sns.boxplot(y='Score', data=df1)


# In[55]:


sns.boxplot(y='Weigh', data=df1)


# In[56]:


sns.boxplot(data=df1)


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Q9=pd.read_csv("C:/Users/A/Downloads/Q9_a.csv")


# In[3]:


Q9.head(10)


# In[6]:


plt.hist(Q9['speed']) 


# In[7]:


plt.hist(Q9['dist']) 


# In[8]:


plt.boxplot(Q9['speed'],vert = True)


# In[9]:


sns.boxplot(data=Q9)


# In[15]:


plt.boxplot(Q9)


# In[2]:


from scipy import stats
import pandas as pd
import numpy as np


# In[3]:


stats.norm.ppf(0.97)


# In[4]:


stats.norm.ppf(0.99)


# In[5]:


stats.norm.ppf(0.98)


# In[6]:


Q12=pd.read_csv("C:/Users/A/Downloads/Q12.csv")


# In[7]:


Q12


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


plt.hist(Q12['Score']) 


# In[1]:


# question no 20
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from scipy import stats
from scipy.stats import norm


# In[3]:


cars=pd.read_csv("C:/Users/A/Downloads/cars.csv")


# In[4]:


cars


# In[27]:


#p(mpg>38)
1-stats.norm.cdf(38,cars.MPG.mean(),cars.MPG.std())


# In[22]:


#mpg<40
stats.norm.cdf(40,cars.MPG.mean(),cars.MPG.std())


# In[21]:


#20<mpg<50
stats.norm.cdf(0.50,cars.MPG.mean(),cars.MPG.std())-stats.norm.cdf(0.20,cars.MPG.mean(),cars.MPG.std())


# In[30]:


stats.norm.cdf(50,34.422,9.13144)-(1-stats.norm.cdf(20,34.422,9.13144))


# In[31]:


#20<mpg<50
stats.norm.cdf(50,cars.MPG.mean(),cars.MPG.std())-(1-stats.norm.cdf(20,cars.MPG.mean(),cars.MPG.std()))


# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


from numpy.random import seed
from numpy.random import randn
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot


# In[59]:


cars=pd.read_csv("C:/Users/A/Downloads/cars.csv")


# In[60]:


cars[0:5]


# In[61]:


cars = cars[['MPG']]


# In[62]:


cars


# In[66]:


qqplot(cars, line='45')
pyplot.show()


# In[67]:


from numpy.random import seed
from numpy.random import randn
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot


# In[68]:


wc=pd.read_csv("C:/Users/A/Downloads/wc_at.csv")


# In[69]:


wc


# In[70]:


wc = wc[['Waist']]


# In[71]:


wc


# In[90]:


wc=pd.read_csv("C:/Users/A/Downloads/wc_at.csv")


# In[91]:


wc


# In[89]:


import statsmodels.api as smf
import pylab as py 


# In[92]:


smf.qqplot(wc['Waist'],line='s')
py.show() 


# In[76]:


wc=pd.read_csv("C:/Users/A/Downloads/wc_at.csv")


# In[ ]:





# In[93]:


qqplot(wc["AT"], line='s')
pyplot.show()


# In[80]:


from scipy import stats
import pandas as pd
import numpy as np


# In[81]:


stats.norm.ppf(0.95)


# In[82]:


stats.norm.ppf(0.97)


# In[4]:


stats.norm.ppf(0.80)


# In[7]:


stats.t.ppf(0.975,df=24)


# In[10]:


stats.t.ppf(0.98,df=24)


# In[11]:


stats.t.ppf(0.995,df=24)


# In[34]:


AVG_WGT3=stats.norm.interval(0.98, loc = 200, scale = 30)


# In[35]:


np.round(AVG_WGT3, 3)


# In[36]:


AVG_WGT1 = stats.norm.interval(0.97, loc = 200, scale = 30)


# In[37]:


np.round(AVG_WGT1, 3)


# In[38]:


AVG_WGT2=stats.norm.interval(0.99, loc = 200, scale = 30)


# In[39]:


np.round(AVG_WGT2, 3)


# In[42]:


np.round((1-stats.norm.cdf(20, loc = df4.MPG.mean(), scale = df4.MPG.std()))-(stats.norm.cdf(40, df4.MPG.mean(), scale = df4.MPG.std())) , 3))


# In[43]:


cars=pd.read_csv("C:/Users/A/Downloads/cars.csv")


# In[44]:


cars


# In[45]:


np.round((1-stats.norm.cdf(20, loc = cars.MPG.mean(), scale = cars.MPG.std()))-(stats.norm.cdf(40, cars.MPG.mean(), scale = cars.MPG.std())) , 3)


# In[52]:


import statsmodels.api as smf
import pylab as py 


# In[83]:


smf.qqplot(cars["MPG"],line='s')
py.show() 


# In[94]:


print('Z scores at 90% confidence interval is', np.round(stats.norm.ppf(.95), 2))


# In[95]:


print('Z scores at 94% confidence interval is', np.round(stats.norm.ppf(.97), 2))


# In[96]:


print('Z scores at 60% confidence interval is', np.round(stats.norm.ppf(.80), 2))


# In[97]:


print(' t scores at 95% confidence interval is', np.round(stats.t.ppf(0.975, df = 24), 2))


# In[98]:


print(' t scores at 96% confidence interval is', np.round(stats.t.ppf(0.98, df = 24), 2))


# In[99]:


print(' t scores at 99% confidence interval is', np.round(stats.t.ppf(0.995, df = 24), 2))


# In[104]:


t_value = (260 - 270)/(90/np.sqrt(18))


# In[105]:


print('critical value = ', np.round(t_value, 2))


# In[106]:


print('probabilty for average life of no more than 260 days is', np.round(stats.t.cdf(t_value, df=17), 2))


# In[107]:


import numpy as np
from scipy import stats
from scipy.stats import norm


# In[108]:


Mean = 5+7
print('Mean Profit is Rs', Mean*45,'Million')


# In[109]:


SD = (3^2)+(4^2)
print('Standard Deviation is Rs', SD*45, 'Million')


# In[110]:


print('Range is Rs',(stats.norm.interval(0.95,540,315)),'in Millions')


# In[111]:


X= 540+(-1.64)*(315)
print('5th percentile of profit (in Million Rupees) is',np.round(X,2))


# In[112]:


stats.norm.cdf(0,5,3)


# In[113]:


stats.norm.cdf(0,7,4)


# In[ ]:




