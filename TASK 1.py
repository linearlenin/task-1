#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')


# In[51]:


a = pd.read_csv('Iris.csv')
a


# In[52]:


a.info()


# In[53]:


a.describe()


# In[54]:


a.isnull()


# In[55]:


a.isnull().sum()


# In[58]:


a.boxplot()


# In[24]:



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[27]:


x = a[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = a['Species']


# In[28]:


x.head()


# In[29]:


y.head()


# In[31]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=70, random_state=30)


# In[35]:


x_train ,x_test


# In[36]:


y_train,y_test


# In[38]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[40]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)


# In[42]:


y_pred = knn.predict(X_test)
y_pred


# In[44]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[46]:


classification_rep = classification_report(y_test, y_pred, target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
classification_rep


# In[48]:


confusion_mat = confusion_matrix(y_test, y_pred)
confusion_mat


# In[ ]:





# In[ ]:





# In[ ]:




