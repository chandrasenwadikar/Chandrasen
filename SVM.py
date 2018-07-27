# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 13:26:04 2018

@author: Chandrasen
"""



import numpy as np # libraries for array operations
import pandas as pd # for data handling
from sklearn import  preprocessing #data sampling,model and preprocessing 

path ="E:\Class\Data\Titanic_train.csv"
#g_path = "./"
titanic_df = pd.read_csv(path) # read Data

titanic_df.head() # print few data


""" Data exploration and processing"""
titanic_df['Survived'].mean()


titanic_df.groupby('Pclass').mean()

class_sex_grouping = titanic_df.groupby(['Pclass','Sex']).mean()
class_sex_grouping

class_sex_grouping['Survived'].plot.bar()       


group_by_age = pd.cut(titanic_df["Age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()     

titanic_df.count()

titanic_df = titanic_df.drop(['Cabin'], axis=1)   

 
titanic_df = titanic_df.dropna()    

titanic_df.count()

""" Data preprocessing function"""
def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['PassengerId','Name','Ticket'],axis=1)
    return processed_df
    

processed_df = preprocess_titanic_df(titanic_df)


X = processed_df.drop(['Survived'], axis=1).values # Features dataset
y = processed_df['Survived'].values # Target variable

#Train Test split
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
pred1= svc.predict(X_test)
print(accuracy_score(y_test,pred1))

from sklearn.grid_search import GridSearchCV
params = [
              {'C': [1,2,3], 'kernel': ['linear']}, 
              {'C': [10], 'gamma': [0.001], 'kernel': ['rbf']}
         ]

         
grid_svc = GridSearchCV(estimator=rf, param_grid=params,cv =3)
grid_svc.fit(X_train , y_train )

final_model = grid_svc.best_estimator_

final_model.fit(X_train , y_train)

pred_cv = final_model.predict(X_test)
print(accuracy_score(y_test,pred_cv))

print(grid_svc.best_score_)
print(grid_svc.best_params_)
final_SVM = SVC(C =1,kernel  = "linear")


#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred= lr.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))

from sklearn.model_selection import cross_val_score as cv
cv_score = cv(lr, X_train, y_train, cv=5)

