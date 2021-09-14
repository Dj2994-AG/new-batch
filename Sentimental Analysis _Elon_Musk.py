#!/usr/bin/env python
# coding: utf-8

# In[83]:


from __future__ import print_function


# In[84]:


import numpy as np 
import pandas as pd 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

import warnings
warnings.filterwarnings("ignore")


# In[85]:


tweets=pd.read_csv("C:/Users/A/Downloads/Elon_musk.csv",encoding = "ISO-8859-1", index_col=[0])
tweets.shape
tweets.head()


# In[86]:


def clean(x):
    #Remove Html  
    x=BeautifulSoup(x).get_text()
    
    #Remove Non-Letters
    x=re.sub('[^a-zA-Z]',' ',x)
    
    #Convert to lower_case and split
    x=x.lower().split()
    
    #Remove stopwords
    stop=set(stopwords.words('english'))
    words=[w for w in x if not w in stop]
    
    #join the words back into one string
    return(' '.join(words))

tweets.Text=tweets.Text.apply(lambda x:clean(x))
tweets.head()


# In[87]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *


# In[88]:


nltk.download('vader_lexicon')


# In[89]:


from nltk import tokenize


# In[90]:


sid = SentimentIntensityAnalyzer()


# In[91]:


tweets['sentiment_compound_polarity']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['compound'])
tweets['sentiment_neutral']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['neu'])
tweets['sentiment_negative']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['neg'])
tweets['sentiment_positive']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['pos'])
tweets['sentiment_type']=''
tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets.shape
tweets.head()


# In[92]:


get_ipython().system('pip install wordcloud')


# In[93]:


tweets=pd.read_csv("C:/Users/A/Downloads/Elon_musk.csv",encoding = "ISO-8859-1")
tweets.shape
tweets.head()


# In[94]:


tweets = tweets.drop(['Unnamed: 0'], axis =1)

tweets.head()


# In[95]:


# Data Cleanse #
tweets['Text'] = tweets['Text'].str.replace(r'http\S+', " ", case=False) # Remove HTML
tweets['Text'] = tweets['Text'].str.replace('RT @[\w]*', " ", case=False) 
tweets['Text'] = tweets['Text'].str.replace('&gt', " ", case=False)
tweets['Text'] = tweets['Text'].str.replace("[^a-zA-Z#]", " ", case=False) #Non-letters
tweets['Text'] = tweets['Text'].str.replace("www", " ", case=False)


# In[96]:


# Test - Remove words < 3 characters (to remove stop words)#
tweets2 = tweets.copy()
tweets2['CText'] = tweets2['Text'].apply(lambda x: ' '.join([w for w in x.split() if
                                                           len(w)>3]))


# In[97]:


#Tokenizer#
tokens = tweets2['CText'].apply(lambda x: x.split())
tokens.head()


# In[98]:


tweets2.tail()


# In[99]:


#Remove common words + stop words - NLTK #
# play, player, played, plays --> play #

stop_words = set(stopwords.words('english'))
from nltk.stem.porter import *
stemmer = PorterStemmer() #stemming
tokens = tokens.apply(lambda x: [stemmer.stem(i)
                                         for i in x])

for i in range(len(tokens)):
    tokens[i] = ' '.join(tokens[i])

tokens2 = []
for w in tokens:
    if w not in stop_words:
        tokens2.append(w)
        
tweets2['Tokens'] = tokens
tweets2['Text'].replace('', np.nan, inplace=True)
tweets2.dropna(subset=['Text'], inplace=True)
tweets2.head()


# In[108]:


get_ipython().system('pip install plotly')


# In[109]:


get_ipython().system('pip install wordcloud')


# In[111]:


get_ipython().system('pip install textblob')


# In[112]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import seaborn as sns
import math
import nltk
import csv
import sklearn
import plotly.offline as py 
import plotly.graph_objs as go
import textblob

from wordcloud import WordCloud
from textblob import TextBlob

from matplotlib import pyplot


# In[113]:


all_words = ' '.join([text for text in tweets2['Tokens']])
wordcloud = WordCloud(width=800, 
                      height=600, 
                      random_state=21, 
                      max_font_size=110).generate(all_words)

plt.figure(figsize=(20, 16))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[115]:


def sentiment(x):
    sentValue = TextBlob(x)
    return sentValue.sentiment.polarity

tweets2['sentiment'] = tweets2['Text'].apply(sentiment)

conditions = [
    (tweets2['sentiment'] > 0 ), #Positive
    (tweets2['sentiment'] < 0),  #Negative
    (tweets2['sentiment'] == 0)]  #Neutral

choices = ['positive', 'negative', 'neutral']
tweets2['Pol_Name'] = np.select(conditions, choices, default=' ')
#tweets2


# In[116]:


#Total Stats#
print('total tweets', len(tweets2))
print('positive tweets', sum(tweets2['Pol_Name'] == 'positive')/len(tweets2)*100, '%')
print('negative tweets', sum(tweets2['Pol_Name'] == 'negative')/len(tweets2)*100, '%')
print('neutral tweets',sum(tweets2['Pol_Name'] == 'neutral')/len(tweets2)*100, '%')


# In[ ]:




