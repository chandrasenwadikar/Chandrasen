
# coding: utf-8

# # Action Analytics 

# ### Step 1 : Importing Data 

# In[367]:


# suppressing warnings
import warnings 
warnings.filterwarnings("ignore")


# In[368]:


# Importing the pandas and numpy

import pandas as pd
import numpy as np


# ###  Step 2 : Inspecting the DataFrame

# In[369]:


# Imporitng data set

action_analytics = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/action_analytics.csv")
action_analytics.head()


# In[370]:


# Check the dimesions of the dataframe
action_analytics.shape


# In[371]:


# Look at the Statistical aspects of the DataFrame
action_analytics.describe()


# In[372]:


# Check the info of each column
action_analytics.info()


# In[373]:


for col in ['CUSTOMER_NUMBER', 'DPD_OF_ACTION']:
    action_analytics[col] = action_analytics[col].astype('object')


# In[374]:


# Converting date from object to numeric value
action_analytics['ACTION_DATE'] = pd.to_numeric(action_analytics.ACTION_DATE.str.replace('-',''))


# In[375]:


action_analytics.info()


# In[376]:


# Checking null values in DataFrame

action_analytics.isnull().sum()


# In[377]:


# Printing the data types of the data frame columns
print (action_analytics.dtypes)


# In[378]:


action_analytics.head()


# In[32]:


# Make the datatypes as DateTime

#action_analytics['ACTION_DATE'] = pd.to_datetime(action_analytics['ACTION_DATE'])


# In[93]:


# Converting date into Integer for checking the impact on Feature Selection
#action_analytics['ACTION_DATE'] = action_analytics['ACTION_DATE'].str.replace('-','').apply(int)


# In[183]:


#action_analytics.CUSTOMER_SEGMENT = pd.to_numeric(action_analytics.CUSTOMER_SEGMENT, errors = 'coerce')


# ### Step 3. Data Preparaton

# In[379]:


# Checking the outliers
aa_act = action_analytics[['CUSTOMER_SEGMENT','ACTION_STRATEGY','DPD_OF_ACTION','COLLECTOR_CODE']]


# In[380]:


# Checking outliers at 25%,50%,75%,90%,95%,99%
aa_act.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# In[381]:


# Assigning the numeric value for each DPD Action record.
varlist = ['DPD_OF_ACTION']

# defining tthe map function
def binary_map(x):
    return x.map({'1-10':1,"11-20":2,"21-30":3,"31-40":4,"41-50":5})

# Applying the function to the list

action_analytics[varlist] = action_analytics[varlist].apply(binary_map)




# In[382]:


# Replace common value of COLLECTOR_CODE with space.
action_analytics.COLLECTOR_CODE = action_analytics.COLLECTOR_CODE.str.replace("[C]", " ")


# In[383]:


action_analytics.ACTION_STRATEGY = action_analytics.ACTION_STRATEGY.str.replace('ACT', '')


# In[384]:


# Defining numeric value to different Customer Segment as defined
varlist1 = ['CUSTOMER_SEGMENT']

# defining tthe map function
def binary_map(x):
    return x.map({'S1':1,"S2":2,"S3":3,"S4":4,"S5":5})

# Applying the function to the list

action_analytics[varlist1] = action_analytics[varlist1].apply(binary_map)



# In[385]:


action_analytics.head()


# ### Step 4 : Test-Train Split

# In[386]:


from sklearn.model_selection import train_test_split


# In[397]:


# Putting feature variable

X= action_analytics.drop(['CUSTOMER_NUMBER','SUCCESSFUL_ACTION'],axis=1)
X.head()


# In[398]:


# Putting response variable to y

y= action_analytics['SUCCESSFUL_ACTION']
y.head()


# In[399]:


# Splitting the data into train and test

X_train, X_test, y_train,y_test = train_test_split(X,y, train_size=0.7,random_state=100)


# ### Spte 5 : Feature Scaling 

# In[400]:


from sklearn.preprocessing import StandardScaler


# In[401]:


scaler = StandardScaler()
X_train[['CUSTOMER_SEGMENT','ACTION_STRATEGY','DPD_OF_ACTION','ACCOUNT_NUMBER','COLLECTOR_CODE']] = scaler.fit_transform(X_train[['CUSTOMER_SEGMENT', 'ACTION_STRATEGY','DPD_OF_ACTION','ACCOUNT_NUMBER','COLLECTOR_CODE']])
X_train.head()


# In[402]:


### Checking the Success Ratio
SR = (sum(action_analytics['SUCCESSFUL_ACTION'])/len(action_analytics['SUCCESSFUL_ACTION'].index))*100
SR


# ### Step 6 : Looking at Correlations

# In[403]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[404]:


# Check the Correlation Matrix
plt.figure(figsize = (20,10))
sns.heatmap(action_analytics.corr(),annot = True)
plt.show()


# ###  Step 7 : Model Building and Feature Scaling using RFE

# In[405]:


import statsmodels.api as sm


# In[406]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[407]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[408]:


from sklearn.feature_selection import RFE


# In[409]:


rfe = RFE(logreg,6)


# In[410]:


rfe = rfe.fit(X_train,y_train)


# In[411]:


rfe.support_


# In[412]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[418]:


col = X_train.columns[rfe.support_]


# In[419]:


X_train.columns[~rfe.support_]


# #### Assigning the model with StatsModel

# In[416]:


X_train_sm = sm.add_constant(X_train[col])


# In[417]:


logm2 = sm.GLM(y_train,X_train_sm, family=sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[420]:


# Getting the preodicated values on the train set

y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[421]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# #### Creating data frame with the actual success action and the predicated probabilities

# In[467]:


y_train_pred_final = pd.DataFrame({'SUCCESSFUL_ACTION':y_train.values, 'SUCCESS_PROB':y_train_pred})
y_train_pred_final['ACTION_STRATEGY'] = y_train.index
y_train_pred_final.head()


# #### Creating new column 'Predicateed with 1 if SUCCESS_PROB>0.5 else 0

# In[468]:


y_train_pred_final['Prediced'] = y_train_pred_final.SUCCESS_PROB.map(lambda x:1 if x > 0.5 else 0)


# In[469]:


# Let'see the output
y_train_pred_final.head()


# ### Confusion Matrix

# In[470]:


from sklearn import metrics


# In[471]:


confusion = metrics.confusion_matrix(y_train_pred_final.SUCCESSFUL_ACTION, y_train_pred_final.Prediced)
print(confusion)


# In[ ]:


# Predicted     No_Successful_Action    SUCCESSFUL_ACTION
# Actual
# No_Successful_Action        344         347
# SUCCESSFUL_ACTION            202       507  


# In[472]:


# Let's check the overll accurcy
print(metrics.accuracy_score(y_train_pred_final.SUCCESSFUL_ACTION, y_train_pred_final.Prediced))


# ### Checking Variance Inflation Factor (VIF)

# In[473]:


# Check for the VIF values of the feature variables
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[474]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

