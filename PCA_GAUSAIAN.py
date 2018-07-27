# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:33:44 2018

@author: DH384961
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
 
#Scaling the values
X = scale(X)

pca = PCA(n_components=4)

pca.fit(X)

#The amount of variance that each PC explains
var= pca.explained_variance_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_, decimals=4)*100)/sum(var)


plt.plot(var1)
plt.xlim(0,5)
plt.show()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)
predictions_prob = clf.predict_proba(X)
predictions  = clf.predict(X)


# code to create validation set
import pandas as pd
from sklearn.cross_validation import train_test_split
data = pd.read_csv("Data/CreditRisk.csv")
X_train, X_test, y_train, y_test = train_test_split( data,data.Loan_Status, test_size=0.25, random_state=42)
X_train.to_csv("Data/Filtered_CreditRisk.csv",index=False)
X_test.to_csv("Data/Validation_CreditRisk.csv",index=False)
