# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:26:27 2018

@author: Admin
"""
print('first program')

1+3

4**4 ##"Exponential command"

var1=50
print(var1)

type(var1) ##Type is used to checked the data type of Var1

Var2='Hello'
print(Var2)

type(Var2)

## DATA TYPES###

##NUMERICAL DATA

i_int=10; print(i_int)

i_float=10.9; print(i_float)
type(i_float)


a="54.5555"
print(a)
type(a)

print(int(float(a))) # converted string to float

##STRINGS

str="Hello World"
print(str) #prints complete string
print(str[0]) #prrints first character of the string
print(str[2:4]) # prints characters starting from 2 to 4
print (str[2:]) #prints string starting from 3rd character
print (str*2) # prints strings two times
print (str + " ", "TEST") # prints concatenated string

#LIST

eg_list=['hello',2,3,'world',9,12]
print(eg_list) # print the full string
print (eg_list[0]) #  print first element of the list
print (eg_list[1:3]) #print element starting from 2nd till 3rd
print (eg_list[2:]) # prints elements starting from 3rd


another_list=[999,'test'," ",'done']
print(another_list *2) #prints list two times
print(eg_list+another_list) #prints concatenated lists
another_list[1]='test'
print(another_list)


#TUPLES

eg_tuple=('hello', 2,3,'world')
print(eg_tuple) #print complete tuple
another_tuple=(999,'hello')
print(eg_tuple[0]) #prints first element of the tuple
print(eg_tuple[1:3]) #print elements starting from 2nd till 3rd
print(eg_tuple[2:]) #prints elements starting from 3rd
print(another_tuple *2) #prints two times
print(eg_tuple + another_tuple) # prints concateneationp
another_tuple[1]='world'

# DICTIONARY

eg_dict={}
eg_dict['one'] #'This is one
eg_dict[2] #this is two

print(eg_dict)
print(eg_dict['one'])

print(eg_dict[2])

another_dict={'name' :'imarticus','code':700,'dept':'ML'}
print(another_dict) #prints complete dictionary
print(another_dict.keys()) # prints all the keys
print(another_dict.values()) # prints all the values

print(another_dict.get('name'))


#OPERATORS

#ARITHEMATIC

print(2*3)
print(2+3)
print(2/4)
print(5%2)
print(3**2)
print(5//2)

#COMPARISION

a=5
b=10
print(a==b)
print(a!=b)
print(a>b)
print(a<b)
print(a>=b)
print(a<=b)

#ASSIGNMENT

#BITWISE

a=60
b=20
print(a&b)
print(a|b)
print(a^b)
print(not(a and b))

#MEMBERSHIP

eg_list=['hello',2,3,'world']
print(22 in eg_list)

#IDENTITY

a=1
b=2
print(a is b)


#MODULES
import gensim

import statistics

example_list=[3,2,4,5,6,6,3,3,2,4,9]

import statistics
print(statistics.mean(example_list))

import statistics as s # instated of every time using statistics make alias to them 

print(s.mean(example_list))

from statistics import mean
print(mean(example_list))

from statistics import mean as m
print(m(example_list))

from statistics import mean as m,median as d
print(m(example_list))
print(d(example_list))


#CONDITION
x=5
y=10

if x > y:
    print('x is greter than y')
else:
        print('y is greater than x')
        
x=5
y=10
z=20

if x>y:
    print('X is greater than Y')
elif x>z:#elif used like else if
    print('X is greater than Z')
else:
    print('x is greater than y or z')
    
#FUNCTION : ALWAYS PASSED BY REFERENCE 
    
def example_function():
    print('first function')
z=20+30
print(z)
example_function()

def example_function2():
    print('second function')
    z=20+30
    return z


return_val=example_function2()
print(return_val)
    
def example_function3(num1,num2):
    print('third function')
    z=num1+num2
    return z

return_val=example_function3(20,30)
print(return_val)

return_val=example_function3(num2=30,num1=80)
print(return_val)

def example_function4(num1,num2=2):
    print('forth example')
    z=num1**num2
    return z
return_val=example_function4(5)
print(return_val)

return_val=example_function4(5,3)
print(return_val)


def example_function5(num1):
    print('fifth example')
    z=num1*example_function3(50,100) #CAlling multiple function in one function
    return z
print(z)

print(example_function5(10))

def a(b,c,d):pass
print (a(2,3,4))

print(type(1/2))

nums=set([1,1,2,3,3,3,4])

print(nums)
print(len(nums))

x=4.5
y=2
print(x//y)

a=[1,2,3,None,(),[]]
print(len(a))

x=True
y=False
z=False

if (x or y) and z:
    print("yes")
else:
    print("no")
    
x=True
y=False
z=False

if not x or y:
    print(1)
elif not x or not y and z:
    print(2)
elif not x or y  or not y and x:
    print(3)
else:
    print(4)
    
name="snow storm"
print(name[6:8])

name="snow storm"
#name[5]='X'
name_new=list(name)
print(name_new)
name_new[5]='X'


name="snow strom"
xx= '='.join(name_new)
print(xx)

#########################################################################


#COLLECTION

import collections as c

#COUNTER

########FILE reading and writing #########

#C:\\Users\\Admin\\Desktop\\test.txt -- path where file gets saved

#read complete file

f=open('C:\\Users\\Admin\\Desktop\\test.txt','r') #double slash required

type(f)

text=f.read()
print(text)
f.close()

# Read file line by line

f=open('C:\\Users\\Admin\\Desktop\\test.txt','r')
print(f.readline()) #  in readline() seek also gets initiated 
print(f.readline())
print(f.readline())
print(f.readline())
f.close()

#Read whole file at once by save it line by line as list

f=open('C:\\Users\\Admin\\Desktop\\test.txt','r')
text=f.readlines()
print(text)
f.close()


#WRITE FILE

f=open('C:\\Users\\Admin\\Desktop\\test.txt','w')
f.write('Hello Test')
f.close()

f=open('C:\\Users\\Admin\\Desktop\\test.txt','w')
text=f.readlines()
print(text)
f.close()

# Append mode

f=open('C:\\Users\\Admin\\Desktop\\test.txt','a')
f.write("Hello test again/n")
f.close()


#Better and safer way to use the file, no need to close file explicitely

with open('C:\\Users\\Admin\\Desktop\\test.txt','r') as f:
     read_data=f.read()
     print(read_data)
     
with open('C:\\Users\\Admin\\Desktop\\test.txt','a') as f:
    f.write('Final test')
print(read_data)

# For PAth Usage

import os
pathDir='C:\\Users\\Admin\\Desktop\\'

f=open(os.path.join(pathDir,'test.txt'),'r')


#LOOPS

testList=[1,2,4,5,6,3,4,5,9]
for x in testList:
    print(x)
    
testList=[1,2,4,5,6,3,4,5,9]
for x in testList:
    print(x*2)
    
testList=[1,2,4,5,6,3,4,5,9]
for x in len(testList):# This program throws the error , but just select the len(testList),prg gets run
    print(x)
    
condition=1
while condition < 10:
    print(condition)
    condition+=1

for x in range(1,11):
    print(x)
    
for x in range(1,11,2):
    print(x)
    
testList=[1,2,4,5,6,3,4,5,9]
for x in range(len(testList)):
    print(testList[x])
    
    
testList=[1,2,4,5,6,3,4,5,9]
for x in range(len(testList)):
    testList[x]=testList[x]*2
    print(testList[x])
    
for i,x in enumerate(testList):
    print(i,':',x,':',x*2)
    
#LAMBDA FUNCTIONS
    
def f(x): 
    return x**2
print(f(8))

g=lambda x:x**2

print(g(8))
    
sum=lambda x,y :x+y
sum(1,1)
sum(3,5)

testList=[1,2,4,5,6,3,4,5,9]
for x in testList:
    print(g(x))
    
f=lambda a,b: a if(a>b) else b
f(10,5)
f(50,100)


#COLLECTIONS

import collections as c


#COUNTER- identify unique values and counter for dictionary

eg_list=['red','blue','green','red','blue','blue']

count_list=c.Counter(eg_list)
print(count_list)

list(count_list.elements())

eg_2=c.Counter(cats=4,dogs=8,puppy=13)
print(eg_2)

list(eg_2.elements())


print(c.Counter('abrakadabra').most_commom(3))


#DEQUE - Doble ended queues

from collections import deque
d=deque('ghi')
for elem in d:
    print(elem.upper())
    
d.append('j') # add a new entry to the right side
d.appendleft('f') # add  new entry to left side
d # shows the representation of the deque
d.pop() # return and remove the rightmost item
d.popleft() # return  and removed the leftmost
list(d) # list the contenets of the deque
d[0] # peek at left most item
d[-1] # peek the right most item

list(reversed(d)) # list the contents of a deque in reverse order
'h' in d # search the deque
d.extend('jkl') # add  multiple elements at once
d

d.rotate(1) # right rotation
d

d.rotate(-1) #left rotation
d

deque(reversed(d)) # make a new deque in reverse order

d.clear() # empty the deque

d.pop() # cannot pop from any empty deque

d.extendleft('abc') #extendleft() reverses the inpur order
d

# Ordered Dict
#regular unsorted dictionary

d={'banana':3,'apple':2,'pear':3,'orange':4}

#Dictionary sorted by key
OrderedDict(sorted(d.items(),key=lambda t:t[0]))



#############NUMPY##################Numerical Python

import numpy as np
a=np.array([1,2,3])
print(a)


a= np.array([[1,2,5],[3,4]])
print(a)

a=np.arrange(15).reshape(3,5)
a
a.shape

a.ndim

a.dtype.name

a.itemsize

a.size

a= np.array([1,2,3,4])
a

b = np.array([(1.5,2,3),(4,5,6)])
b
b.dtype.name

np.zeros((3,4))

np.ones((2,3,4),dtype=np.int16)

###ARRAYS

a= np.arange(6)
a

b=np.arange(12).reshape(4,3)
b

print(np.arange(10000))
print(np.arange(10000).reshape(100,100))

a=np.ones((2,3),dtype=int)
b=np.random.random((2,3))
b

a
a*=3
a

b+=a
b

a=np.random.random((2,3))
a

a.sum()

b.min()

b.max()

b.sum(axis=0) #sum of each column

b.min(axis=1) # min of each row

b.cumsum(axis=1) #Cumlative sum along each row

B=np.arange(3)
B

np.exp(B)
np.sqrt(B)

C=np.array([2.,-1.,4.])
np.add(B,C)

data=np.arange(12).reshape(3,4)
ind=data.argmax(axis=0)

data[1][2]=999
ind=data.argmax(axis=0)

####PANDAS######

import pandas as pd
import numpy as np

data=np.array(['a','b','c','d'])

s= pd.Series(data) #make the series
type(s)
print(s) #index for every element
print(data)

#observe the index

s=pd.Series(data,index=[100,200,202,240])
print(s)

data={'a':0.,'b':1.,'c':2.}
s=pd.Series(data)
print(s)

s=pd. Series([1,2,3,4,5],index=['a','b','c','d','e'])

#retrive first element

print (s[0])

print(s[0:3])
print(s['c'])

print(s[['a','c','d']])

print(s['f'])

#Data frame

import pandas as pd
df=pd.DataFrame()
print(df)

#From list

data=[['Hi',3],['Hello',4]]



#Dictionary