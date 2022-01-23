#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("C:\\Users\\91808\\Downloads\\archive (2)\\car data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df['Fuel_Type'].unique())


# In[6]:


#check missing or null values
df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


final_dataset = df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]


# In[10]:


final_dataset.head()


# In[11]:


final_dataset['Current_Year']=2020


# In[12]:


final_dataset.head()


# In[13]:


final_dataset['no_year']=final_dataset['Current_Year']-final_dataset['Year']


# In[14]:


final_dataset.head()


# In[15]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[16]:


final_dataset.head()


# In[17]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)


# In[18]:


final_dataset.head()


# In[19]:


final_dataset = pd.get_dummies(final_dataset,drop_first=True)


# In[20]:


final_dataset.head()


# In[21]:


final_dataset.corr()


# In[22]:


import seaborn as sns


# In[23]:


sns.pairplot(final_dataset)


# In[24]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


#Heat Map
corrmat = final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[26]:


#independent and independent features
X=final_dataset.iloc[:,1:]
Y=final_dataset.iloc[:,0]


# In[27]:


X.head()


# In[28]:


Y.head()


# In[29]:


#features importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,Y)


# In[30]:


print(model.feature_importances_)


# In[31]:


#plot graph of importance features for better visualization
feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[33]:


X_train.shape


# In[34]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()


# In[35]:


#Hyperparameters
import numpy as np
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]
print(n_estimators)


# In[36]:


#Randomized search CV
#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]
#Number of features to consider at every split
max_features = ['auto','sqrt']
#Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5,30,num=6)]
#max_depth.append(None)
#Minimum number of samples required to split a node
min_samples_split = [2,5,10,15,100]
#Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5,10]


# In[37]:


from sklearn.model_selection import RandomizedSearchCV


# In[38]:


#create a random grid
random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
               'max_depth':max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)


# In[39]:


#Use the random grid to search for best hyperparameters
#First create the base model to tune
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42 ,n_jobs=1 )


# In[40]:


rf_random.fit(X_train,Y_train)


# In[41]:


predictions = rf_random.predict(X_test)


# In[42]:


predictions


# In[43]:


sns.distplot(Y_test-predictions)


# In[44]:


plt.scatter(Y_test,predictions)


# In[45]:


import pickle
#open a file , where you are to store the data
file = open('random_forest_regression_model.pkl','wb')
#dump information to that file
pickle.dump(rf_random,file)


# In[ ]:




