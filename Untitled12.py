
# coding: utf-8

# In[5]:


#Basic  plotting 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plotting two 1-D numpy arrays

x = np.linspace(5,100,100)
y = np.linspace(10,1000,100)

plt.plot(x,y)


# In[6]:


# need to call plt.show() explicitly to display the plot

plt.show()


# In[7]:


#can also work with lists through it converts lists to np arrays internally

plt.plot([1,4,6,8],[3,8,3,5])
plt.show()


# In[9]:


# Axis labels and title

plt.plot(x,y)

#x and y labels and title

plt.xlabel ("Current")
plt.ylabel ("Voltage")
plt.title ("Ohm's Law")

#Define the range of labels of the axis
#Arguments  : plt.axis(xmin, xmax, ymin, ymax)

plt.xlim([20,80])
plt.ylim ([200,800])
plt.show()


# In[10]:


#Change the colors and line types

#initialising x and y arrays

x = np.linspace (0,10,20)
y = x*2

#color blue, line type '+'

plt.plot (x,y,'b+ ')

#put x and y lables, and the title

plt.xlabel ("Current")
plt.ylabel ("Voltage")
plt.title ("Ohm's Law")

plt.show()


# In[11]:


#plotting multiple lines on the same plot

x= np.linspace(0,5,10)
y = np.linspace(3,6,10)

#plot three curves : y, y**2 and y **3 with different line types

plt.plot(x,y,'r-',x, y**2,'b+',x,y**3,'g^')
plt.show()


# In[12]:


x= np.linspace(0,5,10)
y = np.linspace(3,6,10)

plt.plot(x, y, 'bo') 
plt.show()


# In[13]:


x= np.linspace(0,5,10)
y = np.linspace(3,6,10)

plt.plot(x, y, 'bO') 
plt.show()


# In[14]:


x= np.linspace(0,5,10)
y = np.linspace(3,6,10)

plt.plot(x, y, 'bcircle') 
plt.show()


# In[15]:


x= np.linspace(0,5,10)
y = np.linspace(3,6,10)

plt.plot(x, y, 'b-circle') 
plt.show()


# In[18]:


# Sub-plots

x = np. linspace(1,10,100)
y = np.log(x)

#inititate new figure explicitly

plt.figure(1)

#create sub plot with 1 row and 2 columns

#Create the first subplot in figure 1

plt.subplot(121) #equivalant to plt.sublopt(1,2,1)
plt.title ("y=log(x)")
plt.plot(x,y)

#create second subplot in figure 1

plt.subplot(122)
plt.title("y=log(x)**2")
plt.plot(x,y**2)
plt.show()


# In[22]:


#Example : create a figure having 4 subplots

x = np. linspace(1,10,100)

#optional command , since matplotlib creates a figure by default anyway

plt.figure(1)

#subplot 2

plt.subplot(2,2,1)
plt.title("Cubic")
plt.plot(x,x**3)

#subplot 1

plt.subplot(2,2,3)
plt.title("Linear")
plt.plot(x,x)

#subplot 3
plt.subplot (2,2,2)
plt.title("Log")
plt.plot(x,np.log(x))

#subplot 4
plt.subplot(2,2,4)
plt.title("Exponential")
plt.plot(x,x**2)

plt.show()


# In[23]:


plt.subplot(4,4,2)
plt.title("test1")
plt.plot(x,x**2)
plt.show()


# In[27]:


plt.subplot(4,4,14)
plt.title("test1")
plt.show()


# In[28]:


plt.subplot(3,3,7)
plt.title("test1")

plt.show()


# In[26]:


plt.subplot(3,3,8)
plt.title("test1")
plt.plot(x,x**2)
plt.show()


# In[29]:


#functionalities of plots

df = pd.read_csv("C:/Users/chandrasen.wadikar/Desktop/market_fact.csv")
df.head()


# In[31]:


#Boxplot : visulaization the distribution of a contineous variables

plt.boxplot(df['Order_Quantity'])
plt.show()


# In[32]:


#Box plot of Sales is quite unreadable , since sales varies
#across a wide range

plt.boxplot(df['Sales'])
plt.show()


# In[33]:


#Range of Sales : min is 2.24 , meadian is 449 , max is 89061

df['Sales'].describe()


# In[35]:


#Usual (Linear) scale subplot

plt.subplot(1,2,1)
plt.boxplot(df['Sales'])

#log scale subplot

plt.subplot(1,2,2)
plt.boxplot(df['Sales'])
plt.yscale('log')
plt.show()


# In[36]:


#Histograms

plt.hist(df['Sales'])
plt.show()


# In[37]:


#The histograms can be made more readable by using a log scale

plt.hist(df['Sales'])
plt.yscale('log')
plt.show()


# In[38]:


#Scatter plot with two variables : Profit and Sales

plt.scatter(df['Sales'], df['Profit'])
plt.show()



# In[39]:


#Reading a PNG image

image = plt.imread("number.png")
plt.imshow(image)
plt.show()

