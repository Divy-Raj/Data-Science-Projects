#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[91]:


df = pd.read_csv("C:\\Users\\91808\\Downloads\\loan-train.csv")


# In[92]:


df.head()


# In[93]:


df.shape


# In[94]:


df.info()


# In[95]:


df.describe()


# In[96]:


pd.crosstab(df['Credit_History'],df['Loan_Status'],margins=True)


# In[97]:


df.boxplot(column='ApplicantIncome')


# In[98]:


df['ApplicantIncome'].hist(bins=20)


# In[99]:


df['CoapplicantIncome'].hist(bins=20)


# In[100]:


df.boxplot(column='ApplicantIncome',by='Education')


# In[101]:


df['LoanAmount'].hist(bins=20)


# In[102]:


df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[103]:


df.isnull().sum()


# In[104]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)


# In[105]:


df['Married'].fillna(df['Married'].mode()[0],inplace=True)


# In[106]:


df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)


# In[107]:


df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)


# In[108]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)


# In[109]:


df['LoanAmount_log'].fillna(df['LoanAmount_log'].mean(),inplace=True)


# In[110]:


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)


# In[111]:


df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)


# In[112]:


df.isnull().sum()


# In[113]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])


# In[114]:


df['TotalIncome_log'].hist(bins=20)


# In[115]:


df.head()


# In[116]:


x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
y = df.iloc[:,12].values


# In[117]:


x


# In[118]:


y


# In[119]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[120]:


print(X_train)


# In[121]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()


# In[122]:


for i in range(0,5):
    X_train[:,i] = labelencoder_X.fit_transform(X_train[:,i])


# In[123]:


X_train[:,7]=  labelencoder_X.fit_transform(X_train[:,7])


# In[124]:


X_train


# In[125]:


labelencoder_Y = LabelEncoder()
y_train=  labelencoder_Y.fit_transform(y_train)


# In[126]:


y_train


# In[127]:


for i in range(0,5):
    X_test[:,i] = labelencoder_X.fit_transform(X_test[:,i])


# In[128]:


X_test[:,7]=  labelencoder_X.fit_transform(X_test[:,7])


# In[129]:


labelencoder_Y = LabelEncoder()
y_test=  labelencoder_Y.fit_transform(y_test)


# In[130]:


X_test


# In[131]:


y_test


# In[132]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# In[133]:


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(criterion='entropy',random_state=0)
DTC.fit(X_train,y_train)


# In[134]:


y_pred = DTC.predict(X_test)
y_pred


# In[135]:


from sklearn import metrics
print('The accuracy of decision tree is:',metrics.accuracy_score(y_pred,y_test))


# In[136]:


from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(X_train,y_train)


# In[137]:


y_pred = NBClassifier.predict(X_test)


# In[138]:


y_pred


# In[139]:


print("The accuracy of Naive Bayes is:",metrics.accuracy_score(y_pred,y_test))


# In[140]:


testdata = pd.read_csv("C:\\Users\\91808\\Downloads\\loan-test.csv")


# In[141]:


testdata.head()


# In[142]:


testdata.info()


# In[143]:


testdata.isnull().sum()


# In[144]:


testdata['Gender'].fillna(testdata['Gender'].mode()[0],inplace=True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0],inplace=True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0],inplace=True)
testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0],inplace=True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0],inplace=True)
testdata['Married'].fillna(testdata['Married'].mode()[0],inplace=True)


# In[145]:


testdata.isnull().sum()


# In[146]:


testdata.boxplot(column='LoanAmount')


# In[147]:


testdata.boxplot(column='ApplicantIncome')


# In[148]:


testdata['LoanAmount'].fillna(testdata['LoanAmount'].mean(),inplace=True)


# In[149]:


testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[150]:


testdata.isnull().sum()


# In[151]:


testdata['TotalIncome']=testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['TotalIncome_log']=np.log(testdata['TotalIncome'])


# In[152]:


testdata.head()


# In[153]:


test = testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[154]:


for i in range(0,5):
    test[:,i]=labelencoder_X.fit_transform(test[:,i])


# In[155]:


test[:,7] = labelencoder_X.fit_transform(test[:,7])


# In[156]:


test


# In[157]:


test = ss.fit_transform(test)


# In[158]:


pred = NBClassifier.predict(test)


# In[159]:


pred

