
# coding: utf-8

# # Action Analytics 

# ### Step 1 : Importing Data 

# In[191]:


# suppressing warnings
import warnings 
warnings.filterwarnings("ignore")


# In[192]:


# Importing the pandas and numpy

import pandas as pd
import numpy as np


# ###  Step 2 : Inspecting the DataFrame

# In[193]:


# Imporitng data set

action_analytics = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/action_analytics.csv")
action_analytics.head()


# In[194]:


# Check the dimesions of the dataframe
action_analytics.shape


# In[195]:


# Look at the Statistical aspects of the DataFrame
action_analytics.describe()


# In[196]:


# Check the info of each column
action_analytics.info()


# In[197]:


for col in ['CUSTOMER_NUMBER', 'DPD_OF_ACTION']:
    action_analytics[col] = action_analytics[col].astype('object')


# In[198]:


action_analytics.info()


# In[199]:


# Checking null values in DataFrame

action_analytics.isnull().sum()


# In[200]:


# Printing the data types of the data frame columns
print (action_analytics.dtypes)


# In[32]:


# Make the datatypes as DateTime

#action_analytics['ACTION_DATE'] = pd.to_datetime(action_analytics['ACTION_DATE'])


# In[93]:


# Converting date into Integer for checking the impact on Feature Selection
#action_analytics['ACTION_DATE'] = action_analytics['ACTION_DATE'].str.replace('-','').apply(int)


# In[183]:


#action_analytics.CUSTOMER_SEGMENT = pd.to_numeric(action_analytics.CUSTOMER_SEGMENT, errors = 'coerce')


# In[201]:


#print (action_analytics.dtypes)
action_analytics.dtypes


# ### Step 3. Data Preparaton

# In[202]:


# Checking the outliers
aa_act = action_analytics[['CUSTOMER_SEGMENT','ACTION_STRATEGY','DPD_OF_ACTION','COLLECTOR_CODE']]


# In[203]:


# Checking outliers at 25%,50%,75%,90%,95%,99%
aa_act.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# In[204]:


# Assigning the numeric value for each DPD Action record.
varlist = ['DPD_OF_ACTION']

# defining tthe map function
def binary_map(x):
    return x.map({'1-10':1,"11-20":2,"21-30":3,"31-40":4,"41-50":5})

# Applying the function to the list

action_analytics[varlist] = action_analytics[varlist].apply(binary_map)


# In[205]:


# Replace common value of COLLECTOR_CODE with space.
action_analytics.COLLECTOR_CODE = action_analytics.COLLECTOR_CODE.str.replace("[C]", " ")


# In[206]:


action_analytics.ACTION_STRATEGY = action_analytics.ACTION_STRATEGY.str.replace('ACT', '')


# In[208]:


# Defining numeric value to different Customer Segment as defined
varlist1 = ['CUSTOMER_SEGMENT']

# defining tthe map function
def binary_map(x):
    return x.map({'S1':1,"S2":2,"S3":3,"S4":4,"S5":5})

# Applying the function to the list

action_analytics[varlist1] = action_analytics[varlist1].apply(binary_map)


# In[209]:


action_analytics.head()


# ### Step 4 : Test-Train Split

# In[210]:


from sklearn.model_selection import train_test_split


# In[211]:


# Putting feature variable

X= action_analytics.drop(['CUSTOMER_NUMBER'],axis=1)
X.head()


# In[212]:


# Putting response variable to y

y= action_analytics['SUCCESSFUL_ACTION']
y.head()


# In[213]:


# Splitting the data into train and test

X_train, X_test, y_train,y_test = train_test_split(X,y, train_size=0.7,random_state=100)


# ### Spte 5 : Feature Scaling 

# In[214]:


from sklearn.preprocessing import StandardScaler


# In[215]:


scaler = StandardScaler()
X_train[['CUSTOMER_SEGMENT','ACCOUNT_NUMBER','ACTION_STRATEGY','DPD_OF_ACTION','COLLECTOR_CODE']] = scaler.fit_transform(X_train[['CUSTOMER_SEGMENT','ACCOUNT_NUMBER','ACTION_STRATEGY','DPD_OF_ACTION','COLLECTOR_CODE']])
X_train.head()


# In[216]:


### Checking the Success Ratio
SR = (sum(action_analytics['SUCCESSFUL_ACTION'])/len(action_analytics['SUCCESSFUL_ACTION'].index))*100
SR


# In[237]:


action_analytics['ACTION_DATE'] = pd.to_datetime(action_analytics['ACTION_DATE'])


# In[238]:


action_analytics.head()


# ### Step 6 : Looking at Correlations

# In[240]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[241]:


# Check the Correlation Matrix
plt.figure(figsize = (20,10))
sns.heatmap(action_analytics.corr(),annot = True)
plt.show()


# ###  Step 7 : Model Building and Feature Scaling using RFE

# In[242]:


import statsmodels.api as sm


# In[247]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[243]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[244]:


from sklearn.feature_selection import RFE


# In[245]:


rfe = RFE(logreg,5)


# In[246]:


rfe = rfe.fit(X_train,y_train)

