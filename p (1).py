#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np


# In[3]:


#Read application_data.csv
dfa = pd.read_csv(r"H:\New folder\casestudyeda\application_data.csv")
dfa.head()


# In[4]:


#Removing columns that are not required 
dfa = dfa.drop(['FLAG_OWN_CAR'], axis = 1)


# In[5]:


dfa = dfa.drop(['DAYS_BIRTH'], axis = 1)


# In[6]:


dfa = dfa.drop(['OWN_CAR_AGE'], axis = 1)


# In[7]:


dfa = dfa.drop(['REG_REGION_NOT_LIVE_REGION'], axis = 1)


# In[8]:


dfa = dfa.drop(['REG_REGION_NOT_WORK_REGION'], axis = 1)


# In[9]:


dfa = dfa.drop(['LIVE_REGION_NOT_WORK_REGION'], axis = 1)


# In[10]:


dfa = dfa.drop(['REG_CITY_NOT_LIVE_CITY'], axis = 1)


# In[11]:


dfa = dfa.drop(['REG_CITY_NOT_WORK_CITY'], axis = 1)


# In[12]:


dfa = dfa.drop(['BASEMENTAREA_AVG'], axis = 1)


# In[13]:


dfa = dfa.drop(['YEARS_BEGINEXPLUATATION_AVG'], axis = 1)


# In[14]:


dfa = dfa.drop(['COMMONAREA_AVG'], axis = 1)


# In[15]:


dfa = dfa.drop(['ELEVATORS_AVG'], axis = 1)


# In[16]:


dfa = dfa.drop(['ENTRANCES_AVG'], axis = 1)


# In[17]:


dfa = dfa.drop(['FLOORSMAX_AVG'], axis = 1)


# In[18]:


dfa = dfa.drop(['FLOORSMIN_AVG'], axis = 1)


# In[19]:


dfa = dfa.drop(['LANDAREA_AVG'], axis = 1)


# In[20]:


dfa = dfa.drop(['LIVINGAPARTMENTS_AVG'], axis = 1)


# In[21]:


dfa = dfa.drop(['LIVINGAREA_AVG'], axis = 1)


# In[22]:


dfa = dfa.drop(['NONLIVINGAPARTMENTS_AVG'], axis = 1)


# In[23]:


dfa = dfa.drop(['NONLIVINGAREA_AVG'], axis = 1)


# In[24]:


dfa = dfa.drop(['APARTMENTS_MODE'], axis = 1)


# In[25]:


dfa = dfa.drop(['BASEMENTAREA_MODE'], axis = 1)


# In[26]:


dfa = dfa.drop(['YEARS_BEGINEXPLUATATION_MODE'], axis = 1)


# In[27]:


dfa = dfa.drop(['YEARS_BUILD_MODE'], axis = 1)


# In[28]:


dfa = dfa.drop(['COMMONAREA_MODE'], axis = 1)


# In[29]:


dfa = dfa.drop(['ENTRANCES_MODE'], axis = 1)


# In[30]:


dfa = dfa.drop(['FLOORSMAX_MODE'], axis = 1)


# In[31]:


dfa = dfa.drop(['FLOORSMIN_MODE'], axis = 1)


# In[32]:


dfa = dfa.drop(['LANDAREA_MODE'], axis = 1)


# In[33]:


dfa = dfa.drop(['LIVINGAPARTMENTS_MODE'], axis = 1)


# In[34]:


dfa = dfa.drop(['LIVINGAREA_MODE'], axis = 1)


# In[35]:


dfa = dfa.drop(['NONLIVINGAPARTMENTS_MODE'], axis = 1)


# In[36]:


dfa = dfa.drop(['NONLIVINGAREA_MODE'], axis = 1)


# In[37]:


dfa = dfa.drop(['APARTMENTS_MEDI'], axis = 1)


# In[38]:


dfa = dfa.drop(['BASEMENTAREA_MEDI'], axis = 1)


# In[39]:


dfa = dfa.drop(['YEARS_BEGINEXPLUATATION_MEDI'], axis = 1)


# In[40]:


dfa = dfa.drop(['YEARS_BUILD_MEDI'], axis = 1)


# In[41]:


dfa = dfa.drop(['COMMONAREA_MEDI'], axis = 1)


# In[42]:


dfa = dfa.drop(['ELEVATORS_MEDI'], axis = 1)


# In[43]:


dfa = dfa.drop(['ENTRANCES_MEDI'], axis = 1)


# In[44]:


dfa = dfa.drop(['FLOORSMAX_MEDI'], axis = 1)


# In[45]:


dfa = dfa.drop(['FLOORSMIN_MEDI'], axis = 1)


# In[46]:


dfa = dfa.drop(['LANDAREA_MEDI'], axis = 1)


# In[47]:


dfa = dfa.drop(['LIVINGAPARTMENTS_MEDI'], axis = 1)


# In[48]:


dfa = dfa.drop(['LIVINGAREA_MEDI'], axis = 1)


# In[49]:


dfa = dfa.drop(['NONLIVINGAPARTMENTS_MEDI'], axis = 1)


# In[50]:


dfa = dfa.drop(['NONLIVINGAREA_MEDI'], axis = 1)


# In[51]:


dfa = dfa.drop(['FONDKAPREMONT_MODE'], axis = 1)


# In[52]:


dfa = dfa.drop(['WALLSMATERIAL_MODE'], axis = 1)


# In[53]:


dfa = dfa.drop(['EMERGENCYSTATE_MODE'], axis = 1)


# In[54]:


dfa = dfa.drop(['OBS_30_CNT_SOCIAL_CIRCLE'], axis = 1)


# In[55]:


dfa = dfa.drop(['DEF_30_CNT_SOCIAL_CIRCLE'], axis = 1)


# In[56]:


dfa = dfa.drop(['OBS_60_CNT_SOCIAL_CIRCLE'], axis = 1)


# In[57]:


dfa = dfa.drop(['DEF_60_CNT_SOCIAL_CIRCLE'], axis = 1)


# In[58]:


dfa = dfa.drop(['FLAG_DOCUMENT_2'], axis = 1)


# In[59]:


dfa = dfa.drop(['FLAG_DOCUMENT_4'], axis = 1)


# In[60]:


dfa = dfa.drop(['FLAG_DOCUMENT_5'], axis = 1)


# In[61]:


dfa = dfa.drop(['FLAG_DOCUMENT_6'], axis = 1)


# In[62]:


dfa = dfa.drop(['FLAG_DOCUMENT_7'], axis = 1)


# In[63]:


dfa = dfa.drop(['FLAG_DOCUMENT_8'], axis = 1)


# In[64]:


dfa = dfa.drop(['FLAG_DOCUMENT_9'], axis = 1)


# In[65]:


dfa = dfa.drop(['FLAG_DOCUMENT_10'], axis = 1)


# In[66]:


dfa = dfa.drop(['FLAG_DOCUMENT_11'], axis = 1)


# In[67]:


dfa = dfa.drop(['FLAG_DOCUMENT_12'], axis = 1)


# In[68]:


dfa = dfa.drop(['FLAG_DOCUMENT_13'], axis = 1)


# In[69]:


dfa = dfa.drop(['FLAG_DOCUMENT_14'], axis = 1)


# In[70]:


dfa = dfa.drop(['FLAG_DOCUMENT_15'], axis = 1)


# In[71]:


dfa = dfa.drop(['FLAG_DOCUMENT_16'], axis = 1)


# In[72]:


dfa = dfa.drop(['FLAG_DOCUMENT_17'], axis = 1)


# In[73]:


dfa = dfa.drop(['FLAG_DOCUMENT_18'], axis = 1)


# In[74]:


dfa = dfa.drop(['FLAG_DOCUMENT_19'], axis = 1)


# In[75]:


dfa = dfa.drop(['FLAG_DOCUMENT_20'], axis = 1)


# In[76]:


dfa = dfa.drop(['FLAG_DOCUMENT_21'], axis = 1)


# In[77]:


dfa = dfa.drop(['AMT_REQ_CREDIT_BUREAU_HOUR'], axis = 1)


# In[78]:


dfa = dfa.drop(['AMT_REQ_CREDIT_BUREAU_DAY'], axis = 1)


# In[79]:


dfa = dfa.drop(['EXT_SOURCE_1'], axis = 1)


# In[80]:


dfa = dfa.drop(['APARTMENTS_AVG'], axis = 1)


# In[81]:


dfa = dfa.drop(['YEARS_BUILD_AVG'], axis = 1)


# In[82]:


dfa = dfa.drop(['ELEVATORS_MODE'], axis = 1)


# In[83]:


dfa = dfa.drop(['HOUSETYPE_MODE'], axis = 1)


# In[84]:


dfa = dfa.drop(['TOTALAREA_MODE'], axis = 1)


# In[85]:


#Replacing null values with median
dfa = dfa.fillna(dfa.median(axis=0))


# In[86]:


#Counting null values
dfa.isnull().sum(axis=0)


# In[87]:


#Read previous_application.csv
dfp = pd.read_csv(r"H:\New folder\casestudyeda\previous_application.csv")
dfp.head()


# In[88]:


#Counting null values
dfp.isnull().sum(axis=0)


# In[89]:


#Replacing null values with median
dfp = dfp.fillna(dfp.median(axis=0))


# In[90]:


#Counting null values
dfp.isnull().sum(axis=0)


# In[91]:


# Merging application and previous_application dataframes
df = pd.merge(dfa,dfp, on ="SK_ID_CURR" )
df.head()


# In[92]:


#Counting null values
df.isnull().sum(axis=0)


# In[93]:


#Removing XNA with null
df = df.replace('XNA', np.nan)


# In[94]:


#Removing XAN with null
df = df.replace('XNA', np.nan)


# In[95]:


#Undersampling

no_frauds = len(df[df['TARGET'] == 1])


# In[96]:


non_fraud_indices = df[df.TARGET == 0].index


# In[97]:


random_indices = np.random.choice(non_fraud_indices,no_frauds, replace=False)


# In[98]:


fraud_indices = df[df.TARGET == 1].index


# In[99]:


under_sample_indices = np.concatenate([fraud_indices,random_indices])


# In[100]:


df1 = df.loc[under_sample_indices]


# In[101]:


#Undersampled Data
df1.head(1)


# In[102]:


#Checking balancing ratio in merged data
zero = (df.TARGET == 0).sum(axis=0)


# In[103]:


one = (df.TARGET == 1).sum(axis=0)


# In[104]:


zero/one


# In[105]:


#Checking balancing ratio in sampled data

zero = (df1.TARGET == 0).sum(axis=0)


# In[106]:


one = (df1.TARGET == 1).sum(axis=0)


# In[107]:


zero/one


# In[108]:


#Univariate Analysis
# Data to perform quantitative Univariate Analysis
dfu1 = df1.select_dtypes(exclude=['object'])
dfu1.head(1)


# In[109]:


#Identifying Outlier
import seaborn as sns

sns.set(style="whitegrid")

ax = sns.boxplot(x=dfu1['AMT_INCOME_TOTAL'])


# In[110]:


ax = sns.boxplot(x=dfu1['AMT_CREDIT_x'])


# In[111]:


ax = sns.boxplot(x=dfu1['CNT_CHILDREN'])


# In[112]:


ax = sns.boxplot(x=dfu1['AMT_ANNUITY_x'])


# In[113]:


ax = sns.boxplot(x=dfu1['DAYS_EMPLOYED'])


# In[114]:


ax = sns.boxplot(x=dfu1['DAYS_REGISTRATION'])


# In[115]:


ax = sns.boxplot(x=dfu1['CNT_FAM_MEMBERS'])


# In[116]:


ax = sns.boxplot(x=dfu1['AMT_APPLICATION'])


# In[117]:


ax = sns.boxplot(x=dfu1['AMT_CREDIT_y'])


# In[118]:


ax = sns.boxplot(x=dfu1['AMT_DOWN_PAYMENT'])


# In[119]:


ax = sns.boxplot(x=dfu1['RATE_DOWN_PAYMENT'])


# In[120]:


ax = sns.boxplot(x=dfu1['CNT_PAYMENT'])


# In[121]:


ax = sns.boxplot(x=dfu1['DAYS_FIRST_DRAWING'])


# In[122]:


ax = sns.boxplot(x=dfu1['DAYS_FIRST_DUE'])


# In[123]:


ax = sns.boxplot(x=dfu1['DAYS_LAST_DUE'])


# In[124]:


#Univariate Analysis
# Data to perform categorical Univariate Analysis
dfu2 = df1.select_dtypes(include = ['object'])
dfu2.head(1)


# In[126]:


#Range frequency graph
import matplotlib.pyplot as plt
for x in dfu2.columns:
    plt.figure()
    plt.set_title = dfu2[x]
    plt.plot(dfu2[x].value_counts())
    plt.show


# In[127]:


#Quantitative summary (median) matrix
for x in dfu1.columns:
    a = dfu1[x].median()
    print(round(a,0))


# In[128]:


df1.head()


# In[129]:


#Bivariate analisis
pd.pivot_table(df1,index=["TARGET"],aggfunc=np.mean)


# In[130]:


#Graph to show Bivariate analisis
for x in dfu1.columns:
    plt.figure()
    plt.plot(df1.TARGET,dfu1[x])
    plt.show


# In[131]:


#Segmenting data with respect to TARGET column
dfv = pd.pivot_table(df1,index=["TARGET"])


# In[132]:


dfv.head()


# In[133]:


#Correlation heat map for segmented data with respect to TARGET column
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,30))      
ax = sns.heatmap(dfv.corr(), linewidths=.5, )

