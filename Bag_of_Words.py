#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer


# In[9]:


vectorizer=CountVectorizer()


# In[11]:


phrase=["The food is delicious and tasty",
        "Everyone liked the food"]


# In[12]:


vectorizer.fit(phrase)


# In[16]:


b= vectorizer.transform(phrase)


# In[17]:


print(b)


# In[19]:


print(b.toarray())


# In[21]:


print(len(vectorizer.vocabulary_))


# In[24]:


print(vectorizer.vocabulary_)


# In[25]:


print(vectorizer.vocabulary_.get("delicious"))


# In[26]:


print(vectorizer.vocabulary_.get("everyone"))


# In[ ]:




