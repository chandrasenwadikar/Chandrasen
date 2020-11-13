# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:26:20 2018

@author: Chandrasen.Wadikar
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import warnings
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

path= pd.read_csv("C:\\Users\\chandrasen.wadikar\\Desktop\\PA data.csv")
path

path.head()

path.rename(columns={path.columns[1]:"Project_Name"})
path.rename(columns={path.columns[2]:"PM_Lead"})
path.rename(columns={path.columns[3]:"Defects_by_PA"})
path.rename(columns={path.columns[4]:"Defects_by_UAT"})
path.rename(columns={path.columns[5]:"Defect_lek"})
path.rename(columns={path.columns[6]:"functional_defects"})
path.rename(columns={path.columns[7]:"Percen_functional_def"})
path.rename(columns={path.columns[8]:"Total_open_defects"})
path.rename(columns={path.columns[9]:"Defect_re_open"})
path.rename(columns={path.columns[10]:"Total_tc"})
path.rename(columns={path.columns[11]:"Time_taken_tc"})
path.rename(columns={path.columns[12]:"Total_tc_exectuted"})
path.rename(columns={path.columns[13]:"Time_taken_exec"})
path.rename(columns={path.columns[14]:"Total_RA"})
path.rename(columns={path.columns[15]:"test_RA"})
#path.rename(columns={path.columns[16]:"RA_Coverage"})

path

path.shape

from sklearn.linear_model import LinearRegression
X = path.drop('Total_RA',axis=1)
lm=LinearRegression
lm

lm.fit()

lm.predict()