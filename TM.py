# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:19:53 2017

@author: DH384961
"""
""" Data set Link: http://mlg.ucd.ie/datasets/bbc.html"""


import os

data_path = "Data/bbc/" # Data Path
directrys = os.listdir(data_path) #
file_list = []  ; file_count =[]
for path,direc,files in os.walk(data_path): 
    file_count.append(len(files)) # appending file counts drom each folder 
    file_list.append(files) # append file names


# Define target variable as news_type
news_type = []; j = 0
for i in file_count[1:]:
    news_type = news_type + [j]*i
    j = j+1

news_mappings = {i:j for i,j in enumerate(directrys)} # create key value dictionary pair for mapping new type to 0,1,2..  

import numpy as np
print(len(news_type))
news_type = np.array(news_type) # Target Variable

#append file content in list format
data = []
j = 1
for i in directrys:
    f_names = file_list[j]
    j = j+1
    for k in f_names:
        f = open(data_path + i+"/" +k) # open the file with path specified
        data.append ( f.read())
        
        
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
T = tfidf() # define tfidf object 
T.fit(data) # fit for your data for td * idf value
data_TFIDF= T.transform(data) # transform to tf*idf array


import re # import regex library

data_clean =[re.sub('[^a-zA-Z]+' , ' ' ,doc ) for doc in data] # For each file have only alphabeticals
data_clean = [doc.lower() for doc in data_clean] # to lower case

from nltk.tokenize import word_tokenize
tokenized_data = [word_tokenize(doc) for doc in data_clean]
print (tokenized_data[0])


# remove stop wors
from nltk.corpus import stopwords
x= stopwords.words('english') # this can be your own list of words
d = []
data_st = []
for doc in tokenized_data:
    for word in doc:
        if word not in x:
            d.append(word)
    data_st.append(d)
    d = []


# stemming and lemmatizing
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
porter = PorterStemmer()
wordnet = WordNetLemmatizer()
d = []
data_stem_lem = []
for doc in data_st:
    for word in doc:
        word = porter.stem(word)
        word = wordnet.lemmatize(word)
        d.append(word)
    data_stem_lem.append(d)
    d = []

# reverse of tokenisation for each document
X =[(" ").join(doc)for doc in data_stem_lem]
T = tfidf() 
T.fit(X)
data_TFIDF= T.transform(X)

# Sparsity reduction
f = []
for i in range(data_TFIDF.shape[1]):
    # removing 99% sparsity column (here removing 99% zero value columns i.e. 1% non zero)
    if (data_TFIDF[:,i].count_nonzero()/data_TFIDF.shape[0]) > 0.01:
        f.append(i)
 
#Final data        
X =data_TFIDF[:,f]
X.shape 

# normal classification process OR clustering if no target defined
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,news_type,test_size=0.2)
 

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100 )
model.fit(X_train,y_train)

predictions = model.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


