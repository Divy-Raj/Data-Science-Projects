#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("C:\\Users\\91808\\Downloads\\weatherAUS.csv")
X = df.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
Y = df.iloc[:,-1].values


# In[3]:


print(X)


# In[4]:


print(Y)


# In[5]:


Y = Y.reshape(-1,1)
print(Y)


# In[6]:


#Dealing with invalid dataset
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)


# In[7]:


print(X)


# In[8]:


print(Y)


# In[9]:


#Encoding Dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
le1 = LabelEncoder()
X[:,4] = le1.fit_transform(X[:,4])
le2 = LabelEncoder()
X[:,6] = le2.fit_transform(X[:,6])
le3 = LabelEncoder()
X[:,7] = le3.fit_transform(X[:,7])
le4 = LabelEncoder()
X[:,-1] = le4.fit_transform(X[:,-1])
le5 = LabelEncoder()
Y = le5.fit_transform(Y)


# In[10]:


print(X)


# In[11]:


print(Y)


# # Feature Scaling

# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[13]:


print(X)


# In[14]:


#Splitting dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[15]:


print(X_train)


# In[16]:


print(Y_train)
#Y_train = Y_train.reshape(1,-1)
#print(Y_train)


# In[17]:


#Training Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,random_state=0)
classifier.fit(X_train,Y_train)


# In[18]:


classifier.score(X_train,Y_train)


# In[19]:


y_pred = classifier.predict(X_test)


# In[20]:


print(y_pred)


# In[21]:


y_pred = le5.inverse_transform(y_pred)


# In[22]:


print(y_pred)


# In[23]:


print(Y_test)


# In[24]:


Y_test = le5.inverse_transform(Y_test)


# In[25]:


print(Y_test)


# In[26]:


Y_test = Y_test.reshape(-1,1)
y_pred = y_pred.reshape(-1,1)


# In[30]:


df1 = np.concatenate((Y_test,y_pred),axis = 1)#axis = 1 means it's vertical column
dataframe = pd.DataFrame(df1,columns=['Rain on Tommorrow','Prediction of Rain'])


# In[32]:


print(df1)


# In[33]:


print(dataframe)


# In[35]:


#calculating accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)


# In[38]:


dataframe.to_csv('Prediction.csv')


# In[ ]:




