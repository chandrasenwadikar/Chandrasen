# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:14:20 2018

@author: MA382665
"""

# print
print('first eg')

print("second eg")

print 'xx'


# maths

1+3
4**4


# variable
var4 = 50.0
print(var4)
type(var4)

var2 = 'Hey!'
print(var2)
type(var2)



var3 = "Hey!"
print(var3)


## Data Types

## Numerical
i_int = 10; print(i_int)
i_int = -10; print(i_int)
type(i_int)

i_float = 10.0
print(i_float)
type(i_float)

i_float = -10.9; print(i_float)

a = "545.2222"
print(float(a))
print(int(float(a)))

## strings
str = 'Hello World!'

print (str)          # Prints complete string
print (str[0])       # Prints first character of the string
print (str[2:5])     # Prints characters starting from 3rd to 5th
print (str[2:])      # Prints string starting from 3rd character
print (str * 2)      # Prints string two times
print (str + " TEST") # Prints concatenated string


## lists

eg_list = [ 'hello', 22 , 2.23, 'world', 7.2 ]
print (eg_list)          # Prints complete list
print (eg_list[0])       # Prints first element of the list
print (eg_list[1:3])     # Prints elements starting from 2nd till 3rd
print (eg_list[2:])      # Prints elements starting from 3rd element

another_list = [999, 'hello']
print (another_list * 2)  # Prints list two times
print (eg_list + another_list) # Prints concatenated lists

another_list[1] = 'World'
print (another_list)

## tuples
eg_tuple = ( 'hello', 22 , 2.23, 'world', 7.2 )
another_tuple = (999, 'hello')

print (eg_tuple)           # Prints complete tuple
print (eg_tuple[0])        # Prints first element of the tuple
print (eg_tuple[1:3])      # Prints elements starting from 2nd till 3rd
print (eg_tuple[2:])       # Prints elements starting from 3rd element
print (another_tuple * 2)   # Prints tuple two times
print (eg_tuple + another_tuple) # Prints concatenated tuple

another_tuple[1] = 'World'

## dictionary
eg_dict = {}

eg_dict['one'] = "This is one"

eg_dict[2]     = "This is two"


print(eg_dict)

print (eg_dict['one'])       # Prints value for 'one' key

print (eg_dict[2])           # Prints value for 2 key


another_dict = {'name': 'Learner', 'code':0000, 'dept': 'ML'}
print (another_dict)          # Prints complete dictionary
print (another_dict.keys())   # Prints all the keys
print (another_dict.values()) # Prints all the values

print (another_dict.get('name'))


######
## Operators

# Arithematic
print (2 + 5)

print (5 - 3)

print (2 * 3)

print (5 / 2)

print (5 % 2)

print (5 ** 2)

print (6 // 2)

# Comparison
a = 5
b = 10
print (a==b)
print (a!=b)
print (a>b)
print (a<b)
print (a>=b)
print (a<=b)

# Assignment

a = 5
b = a; print (b)
b += a; print (b) # b = b + a
b -= a; print (b)
b *= a; print (b)
b /= a; print (b)
a = 4
b %= a; print (b)
b **= a; print (b)
b //= a; print (b)


# Bitwise
a = 60
b = 13
print (a&b) # and
print(a|b) # or
print(a^b) # xor
print(~a) # negate


# Logical
a = True
b = False
print (a and b)
print(a or b)
print(not(a and b))


# Membership : test membership in a sequence

eg_list = [ 'hello', 22 , 2.23, 'world', 7.2 ]
print(22 in eg_list)
print('hello' in eg_list)
print(22 not in eg_list)
print('hello world' not in eg_list)


# Identity: compare the memory locations of two objects
a = 1
b = 2
print(a is b)
a = b = 1
print(a is b)

###############################################################################

## modules
example_list = [5,2,5,6,1,2,6,7,2,6,3,5,5]

import statistics
print(statistics.mean(example_list))


import statistics as s

print(s.mean(example_list))


from statistics import mean
print(mean(example_list))

from statistics import mean as m
print(m(example_list))


from statistics import mean as m, median as d

print(m(example_list))
print(d(example_list))





# if
#1
x = 5
y = 10
if x > y:
    print('x is greater than y')
else:
    print('y is greater than x')

#2
x = 5
y = 10
z = 20
if x > y:
    print('x is greater than y')
elif x > z:
    print('x is greater than z')
else:
    print('x is not greater than y or z')


# function : always passed by reference
#1
def example_function():
    print('first function')
    z = 20 + 30
    print(z)

example_function()

#2
def example_function2():
    print('second function')
    z = 20 + 30
    return z


return_val = example_function2()
print(return_val)

#3
def example_function3(num1, num2):
    print('third function')
    z = num1 + num2
    return z

return_val = example_function3(50, 100)
print(return_val)

return_val = example_function3(num2 = 50, num1 = 100)
print(return_val)

#4
def example_function4(num1, num2=2):
    print('fourth function')
    z = num1**num2
    return z

return_val = example_function4(5)
print(return_val)

return_val = example_function4(5, 3)
print(return_val)


def example_function5(num1):
    print('fifth function')
    z = num1*example_function3(50, 100)
    return z

print(z)

print(example_function5(10))
###############################################################################


## for path usage
import os
pathDir = 'C:\\Users\\MA382665\\Desktop\\classes\\ml with python\\'

f = open(os.path.join(pathDir,'first_file.txt'), 'r')



testList = [1,2,4,5,6,3,4,5,9]
for x in testList:
    print(g(x))


# lambda func
def f (x):
    return x**2

print (f(8))

g = lambda x: x**2

print (g(8))

sum = lambda x, y : x + y
sum(1,1)
sum(3,5)

f = lambda a,b: a if (a > b) else b
f(10,5)
f(50,100)




# loop
#1
condition = 1


while condition < 10:
	print(condition)
	condition += 1




#2
testList = [1,2,4,5,6,3,4,5,9]
for x in testList:
    print(x*2)



testList = [1,2,4,5,6,3,4,5,9]
for x in len(testList):
    print(x)



#3
for x in range(1, 11, 2):
    print(x)



testList = [1,2,4,5,6,3,4,5,9]
for x in range(len(testList)):
    testList[x] = testList[x] * 2
    print(testList[x])


for i, x in enumerate(testList):
    print(i, ':', x, ':', x*2)

#####


## collection
import collections as c

# counter
eg_list = ['red', 'blue', 'red', 'green', 'blue', 'blue']

count_list = c.Counter(eg_list)

print(count_list)

list(count_list.elements())

eg_2 = c.Counter(cats=4, dogs=8, puppy=13)
print(eg_2)

list(eg_2.elements())


print(c.Counter('abracadabra').most_common(3))

## deque
from collections import deque
d = deque('ghi')                 # make a new deque with three items

for elem in d:                   # iterate over the deque's elements
    print (elem.upper())




d.append('j')                    # add a new entry to the right side
d.appendleft('f')                # add a new entry to the left side
d                                # show the representation of the deque
d.pop()                          # return and remove the rightmost item

d.popleft()                      # return and remove the leftmost item

list(d)                          # list the contents of the deque
d[0]                             # peek at leftmost item
d[-1]                            # peek at rightmost item


list(reversed(d))                # list the contents of a deque in reverse

'z' in d                         # search the deque

d.extend('jkl')                  # add multiple elements at once
d

d.rotate(1)                      # right rotation
d

d.rotate(-1)                     # left rotation
d

deque(reversed(d))               # make a new deque in reverse order

d.clear()                        # empty the deque
d.pop()                          # cannot pop from an empty deque


d.extendleft('abc')              # extendleft() reverses the input order
d





#############################################
#### File Reading and Writing

# read file at once

f = open('C:\\Users\\MA382665\\Desktop\\classes\\ml_with_python\\first_file.txt', 'r')

type(f)

text = f.read()

print(text)
f.close()

# read file line by line
f = open('first_file.txt', 'r')

print(f.readline())

print(f.readline())

print(f.readline())

f.close()


# read whole file at once by save it line by line as list
f = open('first_file.txt', 'r')
text = f.readlines()
print(text)
f.close()



# write file
f = open('first_file.txt', 'w')
f.write('This is a test.') # return no of characters written
f.close()

f = open('first_file.txt', 'r')
text = f.readlines()
print(text)
f.close()


f = open('first_file.txt', 'a')
f.write('This is a test again./n') # return no of characters written
f.close()



## better and safer way to use file, no need to close file explicitly
with open('first_file.txt', 'r') as f:
    read_data = f.read()
print(read_data)



with open('first_file.txt', 'a') as f:
    f.write('one final time!') # return no of characters written

print(read_data)

#################################################################
## numpy
import numpy as np
a = np.array([1,2,3])
print (a)


a = np.array([[1, 2, 5], [3, 4]])
print (a)


a = np.arange(15).reshape(3, 5)

print(a)

a.shape

a.ndim

a.dtype.name

a.size


a = np.array([1,2,3,4])
print(a)

b = np.array([(1.5,2,3), (4,5,6)])
print(b)
b.dtype.name

np.zeros( (3,4) )


np.ones( (2,3,4), dtype=np.int16 )                # dtype can also be specified


## arrays
a = np.arange(6)   # 1d array
print(a)


b = np.arange(12).reshape(4,3) # 2d array
print(b)


c = np.arange(24).reshape(2,3,4)         # 3d array
print(c)


print(np.arange(10000))

print(np.arange(10000).reshape(100,100))


a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))
print(b)

print(a)
a *= 3
print(a)

b += a
print(b)



a = np.random.random((2,3))
print(a)

a.sum()


b.min()

b.max()

print(b)


b = np.arange(12).reshape(3,4)
print(b)

b.sum(axis=0)                            # sum of each column

b.min(axis=1)                            # min of each row

b.cumsum(axis=1)                         # cumulative sum along each row




B = np.arange(3)
print(B)

print(np.exp(B))

print(np.sqrt(B))

print(C = np.array([2., -1., 4.]))

print(np.add(B, C))


data = np.arange(12).reshape(3,4)

data[1][2] = 999
ind = data.argmax(axis=0)
print(ind)

##########################################################################
## SESSION - 2

##  ordered dict
# regular unsorted dictionary
from collections import OrderedDict
d = {'banana': 3, 'apple': 4, 'pear': 1, 'orange': 2}

# dictionary sorted by key
OrderedDict(sorted(d.items(), key=lambda t: t[0]))
#OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])

# dictionary sorted by value
OrderedDict(sorted(d.items(), key=lambda t: t[1]))
#OrderedDict([('pear', 1), ('orange', 2), ('banana', 3), ('apple', 4)])

# dictionary sorted by length of the key string
OrderedDict(sorted(d.items(), key=lambda t: len(t[0])))
#OrderedDict([('pear', 1), ('apple', 4), ('orange', 2), ('banana', 3)])

## pandas

# series
import pandas as pd
import numpy as np

data = np.array(['a','b','c','d'])

s = pd.Series(data)
type(s)
print (s)
print(data)

## observe the index
s = pd.Series(data,index=[100,101,102,103])
print (s)


data = {'a' : 0., 'b' : 1., 'c' : 2.}
s = pd.Series(data)
print (s)



s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])
#retrieve the first element
print (s[0])

print (s[0:3])

print (s[:3])

print (s[-3:])

print (s['c'])

print (s[['a','c','d']])

print (s['f'])


## dataframe
import pandas as pd
df = pd.DataFrame()
print (df)


# from list
data = [['Hi',2],['Hello',5],['Bye',3]]
df = pd.DataFrame(data,columns=['Word','Len'])
print (df)


# from dict
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
print (df)

#from list of dict
data = [{'a': 1, 'b': 2, 'c': 3}, {'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data,  index=['first', 'second'], columns=['a', 'b', 'c'])
print (df)


## selection
print(df['a'])

print(df['a'][1])

# col add
df['d'] = [4, 30]
print(df)

# col del
df.drop('d', axis = 1)

print(df['a'][0:2])


#append rows
data2 = [{'a': 10, 'b': 20, 'c': 30}, {'a': 50, 'b': 100, 'c': 200}]
df2 = pd.DataFrame(data2,  index=['third', 'fourth'], columns=['a', 'b', 'c'])
print(df2)

df = df.append(df2)
print(df)

df = df.drop('fourth')
print(df)



## dtypes
#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
   'Age':pd.Series([25,26,25,23,30,29,23]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}

#Create a DataFrame
df = pd.DataFrame(d)
print (df.dtypes)

print (df.shape) #shape of the object

print (df.size) # total number of elements

print(df.head()) # first 5 rows

## reading files usind pandas

file_data = pd.read_csv('C:\\Users\\MA382665\\Desktop\\classes\\ml_with_python\\sample_csv.csv')
print(file_data)

type(file_data)



## class
class Student:

   studCount = 0

   def __init__(self, name, marks):
      self.name = name
      self.marks = marks
      Student.studCount += 1

   def displayCount(self):
     print ("Total Students %d" % Student.studCount)

   def displayStudent(self):
      print ("Name : ", self.name,  ", Marks: ", self.marks)

#Creating Instance Objects
"This would create first object of Student class"
stud1 = Student("X", 80)
"This would create second object of Student class"
stud2 = Student("Y", 60)

#Accessing Attributes
stud1.displayStudent()
stud2.displayStudent()
print ("Total Student %d" % Student.studCount)


# exception handling
x = 5
y = 0
try:
    res = x/y
    print(res)
except:
    print('Cannot divide by zero')


x = 5
try:
    res = x/'a'
except ZeroDivisionError:
    print('Cannot divide by zero')
except TypeError:
    print('Type is not correct')
else:
    print(res)


# trying to read a file that doesnt exist
try:
   with open("testfile", "r") as f:
       f.write("This is my test file for exception handling!!")
except IOError:
   print ("Error: can\'t find file or read data")
else:
   print ("Written content in the file successfully")


##  write a file
try:
   with open("testfile", "w") as f:
       f.write("This is my test file for exception handling!!")
except IOError:
   print ("Error: can\'t find file or read data")
else:
   print ("Written content in the file successfully")




## stats


## linear regression

#weight = [63,64,66,69,69,71,71,72,73,75]; print(len(weight))
#
#height = [127,121,142,157,162,156,169,165,181,208]; print(len(height))

weight = np.array([63,64,66,69,69,71,71,72,73,75]) # independent, predictors
print(len(weight))

height = np.array([127,121,142,157,162,156,169,165,181,208]) # dependent
print(len(height))

import matplotlib.pyplot as plt

plt.scatter(weight, height)
plt.xlabel("weight")
plt.ylabel("height")
plt.show()




def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x - n*m_y*m_x)
    SS_xx = np.sum(x*x - n*m_x*m_x)

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return(b_0, b_1)


# estimating coefficients
b_0, b_1 = estimate_coef(weight, height)
print("Estimated coefficients:\nb_0 = {}  \
      \nb_1 = {}".format(b_0, b_1))


reg_line = [(b_0 * x) + b_1 for x in weight]

plt.scatter(weight, height, color = "red")
plt.plot(weight, reg_line)
plt.xlabel("weight")
plt.ylabel("height")
plt.title("Regresion Line")
plt.show()


import math
def rmse(y1, y_hat):
    y_actual = np.array(y1)
    y_pred = np.array(y_hat)
    error = (y_actual - y_pred)**2
    err_mean = round(np.mean(error))
    err_sq = np.sqrt(err_mean)
    return err_sq

print(rmse(height, reg_line))



## on a dataset
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_boston ## imports datasets from scikit-learn


# define the data/predictors as the pre-set feature names
df_x = pd.DataFrame(data = load_boston().data, columns=load_boston().feature_names)
df_x.head()
df_x.shape
list(df_x)

# Put the target (housing value -- MEDV) in another DataFrame
df_y = pd.DataFrame(load_boston().target, columns=["MEDV"])
df_y.head()
df_y.shape


## split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

len(x_test)
len(y_test)
len(x_train)
len(y_train)

### ###
model_lm = linear_model.LinearRegression().fit(x_train,y_train)
model_lm._residues
model_lm.fit_intercept
model_lm.predict

# intercept
print(model_lm.intercept_)

# coefficient
print(model_lm.coef_)

# now pridicting
y_pred_lm = model_lm.predict(x_test)
print(y_pred_lm[0:5]) # print top5 values

print("Orig :  Predicted")
for orig, pred in zip(y_test['MEDV'], y_pred_lm):#ZIP used to take two functions and print 
    print("{}  \t{}".format(orig, pred[0]))

print(rmse(y_test['MEDV'], y_pred_lm))

#Variance
print(model_lm.score(x_test,y_test))

################################
## another and better method
import statsmodels.api as sm
model_OLS = sm.OLS(y_train,x_train).fit()

print(model_OLS.summary())

#prediction
y_pred_OLS = model_OLS.predict(x_test)

print("Orig :  Predicted")
for orig, pred in zip(y_test['MEDV'], y_pred_OLS):
    print("{}  \t{}".format(orig, pred))

print(rmse(y_test['MEDV'], y_pred_OLS))



######CLustering

#####KMEANS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans


x=[1,5,1.5,8,1,9]
y=[2,8,1.8,8,0.6,11]

plt.scatter(x,y)
plt.show()


x=np.array([[1,2],
           [5,8],
           [1.5,1.8],
           [8,8],
           [1,0.6],
           [9,11]])
    
from sklearn.datasets.samples_generator import make_blobs
centers=([1,1],[5,5],[3,10])
X,_=make_blobs(n_samples=500, centers=centers,cluster_std=1)

plt.scatter(x[:,0],x[:,1])
plt.show()

kmeans=KMeans(n_clusters=3)
kmeans.fit(x)

centroids=kmeans.cluster_centers_ #calculate the centroids of each cluster
print(centroids)

labels=kmeans.labels_
print(labels)

colors=['r.','g.','b.','c.','k.','y.','m.']

for i in range (len(x)):
    print("coordinate:",X[i],"label:",labels[i])
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
    
plt.scatter(centroids[:,0],centroids[:,1],
            marker="x",s=150,linewidths =5,zorder =10)

plt.show()





    
