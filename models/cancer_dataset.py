#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
X,y=mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["class 1","class 0"],loc=4)
plt.xlabel('first feature')
plt.ylabel('second feature')


# In[2]:


X.shape


# In[3]:


X,y=mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel('feature')
plt.ylabel('target')


# In[7]:


from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()


# In[8]:


print('print dataset cancer: ',cancer.keys())


# In[10]:


print('The shape of data: ',cancer['data'].shape)


# In[11]:


print('The shape of data: ',cancer.data.shape)


# In[13]:


print('sample counts per class: \n',{n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))})


# In[15]:


np.bincount(cancer.target)


# In[16]:


cancer.feature_names


# In[17]:


cancer_df=pd.DataFrame(cancer.data, columns=cancer.feature_names)


# In[18]:


cancer_df


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target,random_state=0)


# In[22]:


training_accuracy=[]
test_accuracy=[]
neighbors_setting= range(1,11)

for n_neighbors in neighbors_setting:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))


# In[24]:


plt.plot(neighbors_setting,training_accuracy,label='training_accuracy')
plt.plot(neighbors_setting,test_accuracy,label='test_accuracy')
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[ ]:




