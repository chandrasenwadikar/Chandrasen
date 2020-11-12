
# coding: utf-8

# In[3]:


#Plotting Data Distributions
#Univariate variables

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#The commonly used alias for seaborn as sns

import seaborn as sns

# set a seaborn style of your taste

sns.set_style("whitegrid")

#data

df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/market_fact.csv")

#simply density plot

sns.distplot(df['Shipping_Cost'])
plt.show()



# In[9]:


sns.distplot(df['Sales'], bins = 50)
plt.show()


# In[5]:


# rug= True
# plotting only a few points since rug takes a long while

sns.distplot(df['Shipping_Cost'] [:200], rug=True)
plt.show()


# In[7]:


sns.distplot(df['Sales'] [:50],rug=True)
plt.show()


# In[10]:


#Univariate Distributions - Rug Plots

sns.distplot(df['Sales'],hist=False)
plt.show()


# In[11]:


#Subplots

#subplot 1
plt.subplot(2,2,1)
plt.title('Sales')
sns.distplot(df['Sales'])

#Subplot 2

plt.subplot(2,2,2)
plt.title('Profit')
sns.distplot(df['Profit'])

#subplot 3

plt.subplot(2,2,3)
plt.title('Order Quantity')
sns.distplot(df['Order_Quantity'])


#subplot 4
plt.subplot(2,2,4)
plt.title('Shipping Cost')
sns.distplot(df['Shipping_Cost'])

plt.show()



# In[12]:


#Boxplots

sns.boxplot(df['Order_Quantity'])
plt.title('Order Quantity')
plt.show()


# In[13]:


# to plot the values on the vertical axis, specity y= variable
sns.boxplot(y=df['Order_Quantity'])
plt.title('Order Quantity')
plt.show()


# In[14]:


sns.distplot(df['Shipping_Cost'] [:1000], rug=True)
plt.show()

