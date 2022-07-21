#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import mglearn
import numpy as np

mglearn.plots.plot_knn_classification(n_neighbors=1)


X,y=mglearn.datasets.make_forge()
X.shape


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X,y=mglearn.datasets.make_forge()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10)
X_train

mglearn.plots.plot_knn_classification(n_neighbors=3)


from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=19)
clf.fit(X_train,y_train)

print('Test Accuracy:{:.2f}'.format(clf.score(X_test,y_test)))
fig,axes=plt.subplots(1,3,figsize=(10,3))

for n_neighbors, ax in zip([1,3,9],axes):
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=0.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title('{} neighbors(s)'.format(n_neighbors))
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
axes[0].legend(loc=3)


# In[ ]:




