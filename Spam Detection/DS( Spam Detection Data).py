#!/usr/bin/env python
# coding: utf-8

# In[23]:


#import required libraries
import pandas as pd
import string
from nltk.corpus import stopwords


# In[24]:


#Get the spam data collection using pandas
df_spam_collection  = pd.read_csv("C:\\Users\\91808\\Downloads\\NLP(Spam Collection)",sep='\t',names=['response','message'])


# In[25]:


#view first five records 
df_spam_collection.head()


# In[26]:


#view more information about the spam data using describe method
df_spam_collection.describe()


# In[27]:


#view response using group by and describe method
df_spam_collection.groupby('response').describe()


# In[28]:


#verify length of the message and also add it also as a new column (feature)
df_spam_collection['length'] = df_spam_collection['message'].apply(len)


# In[29]:


#view first 5 message with length
df_spam_collection.head()


# In[30]:


#define a function to get rid of stopwords present in the message 
def message_text_process(mess):
    #check characters to see if there are punctuations
    no_punctuation = [char for char in mess if char not in string.punctuation]
    #now form the sentance 
    no_punctuation = ''.join(no_punctuation)
    #Now eliminate any stopwords
    return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]


# In[31]:


#verify that function is working
df_spam_collection['message'].head(5).apply(message_text_process)


# In[32]:


#start text processing with vectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[33]:


#bag of words by applying the function and fit the data (message) into it 
bag_of_words_transformer = CountVectorizer(analyzer=message_text_process).fit(df_spam_collection['message'])


# In[34]:


#print the length of bag of words stored in the vocabulary_attribute
print(len(bag_of_words_transformer.vocabulary_))


# In[35]:


#store bag of words for message using transform method
message_bagofwords = bag_of_words_transformer.transform(df_spam_collection['message'])


# In[36]:


#apply tfidf transformer and fit the bag of words into it (transformed version)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(message_bagofwords)


# In[37]:


#print the shape of the tfidf 
message_tfidf = tfidf_transformer.transform(message_bagofwords)
print(message_tfidf.shape)


# In[38]:


#choose naive bayes model to detect the spam and fit the tfidf data into it
from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB().fit(message_tfidf,df_spam_collection['response'])


# In[39]:


#check the model for the predicted and expected value say for message#2 and message#5['message'][2]
message = df_spam_collection['message'][2]
bag_of_words_for_message = bag_of_words_transformer.transform([message])
tfidf = tfidf_transformer.transform(bag_of_words_for_message)

print('predicted',spam_detection_model.predict(tfidf)[0])
print('expected',df_spam_collection.response[2])


# In[ ]:




