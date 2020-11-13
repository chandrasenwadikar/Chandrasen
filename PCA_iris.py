# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:18:03 2018

@author: DH384961
"""


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from sklearn.preprocessing import scale
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
 
#Scaling the values
#X = scale(X)

pca = PCA()

pca.fit(X)

#The amount of variance that each Principal Components explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


plt.plot(var1)
plt.xlim(0,5)
plt.show()

PCA_data = pca.transform(X)
PCA_data
PCA_data.shape
final_data = PCA_data[:,:2]
final_data.shape

#bbdeepak@yahoo.co.in