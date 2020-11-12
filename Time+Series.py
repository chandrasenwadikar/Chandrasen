
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#set seaborn theme if you prefer

sns.set(style='white')

# Read data
market_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/market_fact.csv")
customer_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/cust_dimen.csv")
product_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/prod_dimen.csv")
shipping_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/shipping_dimen.csv")
orders_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/orders_dimen.csv")

market_df.head()



# In[3]:


# Merging with the orders data to get the Date column

df = pd.merge(market_df,orders_df, how='inner',on='Ord_id')
df.head()


# In[4]:


# now we have Order_Date in the df
# It is stored as string (object) currently

df.info()


# In[5]:


# Convert Order_Date to datetime type

df['Order_Date'] =pd.to_datetime(df['Order_Date'])
                                    
# Order_Date is now datetime type

df.info()


# In[6]:


# agreegating total sales on each day

time_df = df.groupby('Order_Date') ['Sales'].sum()
print(time_df.head())


# In[8]:


#  time series plot
#figure size

plt.figure(figsize=(16,8))

# time series plot

sns.tsplot(data=time_df)
plt.show()


# In[9]:


# extracting month and year from date

# extract month

df['month'] = df ['Order_Date'] .dt.month

# extract year

df['year'] =df['Order_Date'].dt.year

df.head()


# In[10]:


# grouping by year and month

df_time = df.groupby (["year","month"]).Sales.mean()
df_time.head()


# In[13]:


plt.figure(figsize=(8,6))

#time series plot
sns.tsplot(df_time)
plt.xlabel("Time")
plt.ylabel("Sales")
plt.show()


# In[14]:


# Pivoting data using "month"

year_month = pd.pivot_table(df,values='Sales',index='year',columns='month',aggfunc='mean')
year_month.head()


# In[15]:


# Figure size

plt.figure(figsize=(12,8))

# heatmap with a color map of choice

sns.heatmap(year_month,cmap="YlGnBu")
plt.show()

