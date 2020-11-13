# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:02:28 2018

@author: Chandrasen.Wadikar
"""
from rpy2.robjects import r, pandas2ri

def data(iris):
    retun pandas2ri.rispy(r[iris])



df= data(iris)


data()
   
from pydataset import data


from pydataset import data
mtcars=data('mtcars')

load_iris()

from sklearn import datasets

load_mtcars()
load_boston([retun_X_y])

data.head(mtcars)

 from sklearn import datasets
mtcars=datasets.load_mtcars()