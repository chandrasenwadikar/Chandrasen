
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

df=pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/melbourne.csv")
df.head()


# In[4]:


print(df.shape)


# In[5]:


print(df.info())


# In[6]:


#isnull()

df.isnull()


# In[7]:


#Summing up the missing values(column-wise)
df.isnull().sum()


# In[8]:


#columns having at least one missing value
df.isnull().any()


# In[9]:


#Above is equivalant to axis=0 (by default any() operates on columns)

df.isnull().any(axis=0)


# In[11]:


#Check with by tge row
df.isnull().any(axis=1)


# In[12]:


#rows having all missing values
df.isnull().all(axis=1)


# In[13]:


#sum it up to check how many rows have all missing values
df.isnull().all(axis=1).sum()


# In[14]:


#sum of missing values in each row
df.isnull().sum(axis=1)


# In[16]:


#Treating missing values
#summing up the missing values (column-wise)

round(100*(df.isnull().sum()/len(df.index)),2)


# In[21]:


#Removing the three columns

df=df.drop('BuildingArea',axis =1)
df =df.drop('YearBuilt', axis =1)
df = df.drop('CouncilArea', axis =1)

round(100*(df.isnull().sum()/len(df.index)),2)


# In[22]:


#Treating missing values in rows

df[df.isnull().sum(axis=1) > 5]


# In[23]:


#count the number of rows having  >5  missing values
#use len(df.index)

len(df[df.isnull().sum(axis=1)>5].index)


# In[26]:


#4278 rows have more than 5 missing values
#calculate the percentage

100*(len(df[df.isnull().sum(axis=1)>5].index) / len(df.index))


# In[29]:


#retaining the rows having  <=5 NaNs

df= df[df.isnull().sum(axis=1) <=5]

#look at the summary again
round(100*(df.isnull().sum()/len(df.index)),2)


# In[31]:


# removing NaN Price row

df=df[-np.isnan(df['Price'])]

round(100*(df.isnull().sum()/len(df.index)),2)


# In[32]:


df['Landsize'].describe()


# In[36]:


#removing NaNs in Landsize
df= df[-np.isnan(df['Landsize'])]

round(100*(df.isnull().sum()/len(df.index)),2)


# In[37]:


#rows having latitude and longtitude missing
df[np.isnan(df['Lattitude'])]


# In[45]:


df.loc[:,['Lattitude', 'Longtitude']].describe()


# In[46]:


#imputing latitude and longtitude by mean values
df.loc[np.isnan(df['Lattitude']),['Lattitude']] = df ['Lattitude'].mean()
df.loc[np.isnan(df['Longtitude']),['Longtitude']] = df ['Longtitude'].mean()
round(100*(df.isnull().sum()/len(df.index)),2)


# In[47]:


df.loc[:,['Bathroom','Car']].describe()


# In[49]:


#converting to type 'category'
df ['Car'] = df['Car']. astype('category')
#displaying frequncies of each category 
df['Car'].value_counts()


# In[50]:


#imputing NaNs by 2.0
df.loc[pd.isnull(df['Car']),['Car']] =2
round(100*(df.isnull().sum()/len(df.index)),2)


# In[52]:


#converting to type 'category'
df ['Bathroom'] = df['Bathroom']. astype('category')
#displaying frequncies of each category 
df['Bathroom'].value_counts()


# In[54]:


#imputing NaNs by 1
df.loc[pd.isnull(df['Bathroom']),['Bathroom']] =1
round(100*(df.isnull().sum()/len(df.index)),2)


# In[55]:


df.shape


# In[56]:


len(df.index)/23547

