# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:01:38 2018

@author: Chandrasen.Wadikar
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import sklearn
import warnings
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
warnings.simplefilter(action="ignore",category=FutureWarning)
%matplotlib inline

xs=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
ys=[10,12,20,22,21,25,30,21,32,34,35,30,50,45,55,60,66,64,67,72,74,80,79,84]

len(xs),len(ys)

plt.scatter(xs,ys)
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.show()

def slope_intercept(x_val,y_val):
    x=np.array(x_val)
    y=np.array(y_val)
    m=( ( (np.mean(x)*np.mean(y))-np.mean(x*y)) /
       ((np.mean(x)*np.mean(x))-np.mean(x*x)) )
    m=round(m,2)
    b=(np.mean(y)-np.mean(x)*m)
    b=round(b,2)
    
    return m,b

slope_intercept(xs,ys)

m,b=slope_intercept(xs,ys)

reg_line=[(m*x)+b for x in xs]

plt.scatter(xs,ys,color="red")
plt.plot(xs,reg_line)
plt.ylabel("dependent variable")
plt.xlabel("independent variable")
plt.title("Making a regression line")
plt.show()

def rmse(y1,y_hat):
    y_actual=np.array(y1)
    y_pred=np.array(y_hat)
    error=(y_actual-y_pred)**2
    error_mean=round(np.mean(error))
    err_sq=sqrt(error_mean)
    return err_sq

rmse(ys,reg_line)