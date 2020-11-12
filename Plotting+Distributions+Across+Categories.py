
# coding: utf-8

# In[4]:


# Loading libraries and reading the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#set seaborn theme if you prefer

sns.set(style="white")

#read data

market_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/market_fact.csv")
customer_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/cust_dimen.csv")
product_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/prod_dimen.csv")
shipping_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/shipping_dimen.csv")
orders_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/orders_dimen.csv")

#boxplot of a variable

sns.boxplot(y=market_df['Sales'])
plt.yscale('log')
plt.show()


# In[5]:


# merge the dataframe to add a categorical variable

df = pd.merge(market_df,product_df,how ="inner", on="Prod_id")
df.head()


# In[6]:


# boxplot of a vaiable across various categories

sns.boxplot(x='Product_Category', y='Sales', data=df)
plt.yscale('log')
plt.show()


# In[7]:


sns.boxplot(x='Product_Category', y='Sales', data=df)
plt.show()


# In[9]:


df =df[(df.Profit<1000) & (df.Profit>-1000)]

# boxplot of a vaiable across various product categories
sns.boxplot(x='Product_Category',y='Profit',data=df)
plt.show()


# In[10]:


#adjust figure size

plt.figure(figsize=(10,8))

# subplot 1: Sales

plt.subplot(1,2,1)
sns.boxplot(x='Product_Category', y='Sales', data=df)
plt.title('Sales')
plt.yscale('log')

# subplot 2 : Profit
plt.subplot(1,2,2)
sns.boxplot(x='Product_Category',y='Profit', data=df)
plt.title('Proft')
plt.yscale('log')

plt.show()


# In[11]:


# merging with customer df

df= pd.merge(df,customer_df, how="inner", on="Cust_id")
df.head()


# In[13]:


# boxplot of a vairable across various product categories

sns.boxplot(x='Customer_Segment', y='Profit', data=df)
plt.show()


# In[16]:


# set figure size for larger figure

plt.figure(num=None, figsize=(12,8),dpi=80,facecolor='w',edgecolor='k')

#Specify hue= "categoral_varaible"

sns.boxplot(x= 'Customer_Segment',y='Profit',hue="Product_Category",data=df)
plt.show()


# In[18]:


# plot shipping cost as percentage of Sales amount

sns.boxplot(x=df['Product_Category'], y=100*df['Shipping_Cost']/df['Sales'])
plt.ylabel("100*(Shipping Cost/Sales)")
plt.show()


# In[19]:


sns.boxplot(x= 'Product_Category',y='Profit',hue="Customer_Segment",data=df)
plt.show()

