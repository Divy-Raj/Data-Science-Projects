#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:\\Users\\91808\\Downloads\\covid19-ita-regions-latest.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.tail()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# # Relating the variable with scatterplots

# In[8]:


sns.relplot(x="total_confirmed_cases",y="hospitalized_with_symptoms",hue="recovered",data = df)


# In[9]:


sns.pairplot(df)


# In[10]:


sns.relplot(x='total_cases',y='total_hospitalized',kind='line',data=df)


# In[11]:


sns.catplot(x='total_cases',y='recovered',data=df)


# In[ ]:




