#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[2]:


heart_data = pd.read_csv('heart.csv')


# In[3]:


heart_data


# In[4]:


heart_data.head()


# In[5]:


# statistical measures about the data
heart_data.describe()


# In[6]:


x = [2,3,4,5,6,7,8,9]
y = [2,3,4,5,6,7,8,9]
fig, ax = plt.subplots()
ax.plot(x,y);


# In[7]:


fig, ax = plt.subplots()
ax.bar(x,y);


# In[8]:


fig, ax = plt.subplots()
ax.pie(y);


# In[9]:


heart_data['target'].value_counts()


# In[10]:


X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[12]:


model = LogisticRegression()


# In[13]:


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# In[20]:


accuracy_score(X_train_prediction, Y_train)


# In[21]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)


# In[22]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)


# In[23]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)
RandomForestClassifier()


# In[24]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)


# In[25]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)


# In[26]:


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)
# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]== 0):
 print('The Person does not have a Heart Disease')
else:
 print('The Person has Heart Disease')


# In[ ]:





# In[ ]:




