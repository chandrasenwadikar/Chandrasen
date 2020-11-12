
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
#Read data from text file
companies = pd.read_csv ("companies.txt",sep='\t',encoding ="ISO-8859-1" )
companies.head()


# In[3]:


#Getting data from Relational databases

import pymysql

#Create a connection object 'conn'

conn = pymysql.connect(host = "localhost",#your host , local host for your local machine
                      user = "root", #your username, usually root for localhost
                      passwd ='12345', #your password
                      db="information_schema" # name of the database)
                       
#Create a cursor
c= conn.cursor()
                       
# execute a query using c. execute

c.execute ("select * from enginers;")

#getting the first row of data as a tuple

all_rows = c.featchall()
                       
# to get the only first row , use c. featchone() instated featchall()
                       

                       


# In[ ]:


print (type(all_rows)) # it rwturns a tuple of tuples : each row is a tuple
print ([all_rows[:5]]) # printing the first dew rows


# In[ ]:


df = pd.DataFrame(list(all_rows), columns =["engine","support", "comment","transactions","XA", "savepoints"])
df.head()


# In[19]:


#Getting data from Websites
import requests, bs4

#Getting HTML from the Google Play web page

url = "https://play.google.com/store/apps/details?id=com.facebook.orca&hl=en"
req = requests.get(url) 

#Create a bs4 object
#To avoid warrnings, provide "html51lib" explicitly

soup = bs4.BeautifulSoup(req.text, "html5lib")


# In[16]:


#Gettign all the text inside class = "review-body"

reviews =  soup.select ('.review-body')
print(type(reviews))
print(len(reviews))
print('\n')
#Printing an element of the reviews list
print(reviews[6])

