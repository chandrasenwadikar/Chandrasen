
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

market_fact=pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/market_fact.csv")
market_fact.head()


# In[3]:


market_fact.tail()


# In[5]:


market_fact[2:7]


# In[6]:


market_fact[5::2].head()


# In[18]:


#Creating Sub Set of with the provided range 
ss = market_fact[5::2]
ss.head()
#type(ss)


# In[8]:


sales=market_fact['Sales'] # or sales=market_fact.Sales
sales.head()


# In[10]:


print(type(market_fact['Sales']))
print(type(market_fact.Sales))


# In[11]:


#Select columns
market_fact[['Cust_id','Sales','Profit']].head()


# In[12]:


type(market_fact[['Cust_id','Sales','Profit']])


# In[16]:


type(market_fact[['Sales']])


# In[15]:


market_fact[['Sales']]


# In[19]:


#Print only the even numbers of rows of the dataframe

p= market_fact[2: :2]
print(p.head(20))


# In[20]:


#Position based indexing

help(pd.DataFrame.iloc)


# In[21]:


market_fact.iloc[2,4]


# In[22]:


market_fact.iloc[5]


# In[23]:


#The ':' indicates all rows/columns
market_fact.iloc[5,:]


# In[24]:


market_fact.iloc[[3,7,8]]


# In[28]:


market_fact.iloc[[3,7,8], :]


# In[29]:


market_fact.iloc[4:8]


# In[30]:


market_fact.iloc[4:8, :]


# In[31]:


#selecting a single column
market_fact.iloc[:,2]


# In[32]:


#selecing multiple columns

market_fact.iloc[:,3:8]


# In[33]:


# Selecting multiple rows and columns
market_fact.iloc[3:6,2:5]


# In[34]:


#Using booleans

market_fact.iloc[[True, True, False,True,True,False,True]]


# In[38]:


market_fact.iloc[:,[3,4,5,6]]



# #Label Based Indexing
# 
# market_fact.loc[5]
# 

# In[40]:


market_fact.loc[5, :]


# In[41]:


market_fact.loc[[3,7,8]]


# In[42]:


market_fact.loc[[3,7,8],:]


# In[43]:


market_fact.loc[4:8]


# In[44]:


market_fact.loc[2,'Sales']


# In[45]:


market_fact.loc[4:8, ]


# In[46]:


market_fact.loc[4:8,:]


# In[47]:



#To use of label based indexing will be more clear when we have custom row indices
market_fact.set_index('Ord_id',inplace=True)
market_fact.head()


# In[48]:


#select Ord_id = Ord_5406 and some columns
market_fact.loc['Ord_5406',['Sales','Profit','Cust_id']]


# In[51]:


#select multiple orderes using labels, and some columns

market_fact.loc[['Ord_5406','Ord_5446','Ord_5485'],'Sales':'Profit']


# In[52]:


market_fact.loc[[True, True, False,True,True,False,True]]


# In[53]:


#Slicing and Dicing Dataframes
market_fact.Sales>3000 #Series will display


# In[54]:


#Difference of above result and below result , here we have used LOC with condition and beauty is seen in result.
market_fact.loc[market_fact.Sales>3000]


# In[55]:


#You may want to put the ':' to indicate that you want all the columns
#It is more explicit

market_fact.loc[market_fact['Sales']>3000, :]


# In[56]:


#Multiple Conditions 
market_fact.loc[(market_fact.Sales>2000) & (market_fact.Sales<3000) & (market_fact.Profit >100), :]


# In[58]:


#Using OR ('|' ) Operator
market_fact.loc[(market_fact.Sales>2000) | (market_fact.Profit >100),:]


# In[59]:


market_fact.loc[(market_fact.Sales>2000) & (market_fact.Sales<3000) & (market_fact.Profit >100),['Cust_id','Sales','Profit']]


# In[60]:


#You can use == and != opeartor
market_fact.loc[(market_fact.Sales ==4233.15), :]
market_fact.loc[(market_fact.Sales != 1000), :]


# In[61]:


#You may want to select rows whose column value is in an iterable
# For instance, say a colleague gives you a list of customer_ids from a certain region

customers_in_bangalore = ['Cust_1798','Cust_1519','Cust_637','Cust_851']

#to get all the orders from these customers use the isin() function
# It returns a bollean, which you can use to select rows

market_fact.loc[market_fact['Cust_id'].isin(customers_in_bangalore),:]


# In[65]:


market_fact.loc[(market_fact.area>0) & (market_fact.wind>1) & (market_fact.temp >15)]


# In[66]:


#Merging DataFrames
import numpy as np
import pandas as pd

market_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/market_fact.csv")
customer_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/cust_dimen.csv")
product_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/prod_dimen.csv")
shipping_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/shipping_dimen.csv")
orders_df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/orders_dimen.csv")


# In[68]:


market_df.head()


# In[69]:


customer_df.head()


# In[70]:


product_df.head()


# In[71]:


shipping_df.head()


# In[72]:


orders_df.head()


# In[74]:


#Merging Dataframes
#Note that Cust_id is the common column/key, which is provided to the 'on' argument
#how = 'inner' makes sure that only the customer ids present in both dfs are included in result

df_1 = pd.merge(market_df,customer_df, how='inner',on='Cust_id')
df_1.head()


# In[75]:


#Now you can subset the orders made by customers from 'Corporate' Segment
df_1.loc[df_1['Customer_Segment'] =='CORPORATE', :]


# In[78]:


df_2=pd.merge(df_1,product_df,how='inner', on='Prod_id')
df_2.head()


# In[79]:


#select all orders from product category  = office supplies and from the corporate segment

df_2.loc[(df_2['Product_Category'] =='OFFICE SUPPLIES') & (df_2['Customer_Segment'] =='CORPORATE'), :]


# In[82]:


#Merging shipping df
df_3 = pd.merge(df_2,shipping_df, how='inner', on='Ship_id')
df_3.shape


# In[83]:


#Merging the Orders table to create master df

master_df = pd.merge(df_3, orders_df,how='inner', on='Ord_id')
master_df.shape
master_df.head()


# In[85]:


#Concatentating Data Frames
#Dataframes having the same columns
df_1 = pd. DataFrame ({'Name':['Aman','Joy','Juila','Saif'],
                      'Age':['34','31','26','33'],
                      'Gender':['M','M','F','M']}
                     )

df_2 = pd. DataFrame ({'Name':['Akhil','Asha','Preeti'],
                      'Age':['34','31','26'],
                      'Gender':['M','F','F']} 
                     )

df_1


# In[86]:


df_2


# In[87]:


#To concatenate them, once on top of the other, you can use pd.concat
#The first argument is a sequence (list) of dataframes
#axis =0 indicates that we want to concat along the row axis

pd.concat([df_1,df_2],axis=1)


# In[88]:


# A useful and intuitive alternative to concat along the rows is the append() function
# It concatents along the rows
df_1.append(df_2)


# In[93]:


pd.concat([df_1,df_2],axis=0)

