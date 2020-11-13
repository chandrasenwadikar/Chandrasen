# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:54:14 2018

@author: Chandrasen.Wadikar
"""

print("Hello World")

2**2

ein=9
print (ein)
type(ein)

i_int=10; print(i_int)

i_float=10.9 
print(i_float)
type(i_float)

s="897132897183712"
type(s)

print(int(float(s)))

type(s)

c="534.98292"
type(c)

print(int(float(c)))

str="Test World"

print(str[0:3])
print(str[1:6])
print(str[9])
print(str[3:7])
print(str*2)
print(str +"  ",'123')


listv=[1,3,'testing',8,9,0,'by me']
print(listv)
print(listv[2])
print(listv[6:3])
print(listv[4:7])

listnew=[5,6,'validation',9,0,'machine',7,6,'learning']
print(listnew)
print(listnew*2)
print(listv+listnew)
listnew[1]='test'
print(listnew)


extuple=('hello',6,7,'learning')
print(extuple)
another_tuple=(99,'validate')
print(extuple[2:4])
print(extuple[5:9])
print(extuple[0])
print(another_tuple*2)
print(extuple+another_tuple*4)
another_tuple[3]='learning me'

dict1={}
dict1=['one']
dict1=[3]

print(dict1)
print(dict1['one'])

another_dict={'name':'chandrasen','code':23,'code':23,'number':33,'learning':'hello'}
print(another_dict)
print(another_dict.keys())
print(another_dict.values())

print(2*3)
print(2+6)
print(2.3+3.9)
print(4.33-9.88)
print(3/8)

a=10
b=8

print(a==b)
print(a>b)
print(a<b)
print(a>=b)
print(a<=b)
print(a!=b)


a=30
b=90

print(a&b)
print(a|b)
print(a^b)
print(not(a and b))

example_list=['hello',2,3,'test']
print(example_list)
print(22 in example_list)
print('hello' in example_list)

a=1
b=2

print(a is b)

import gensim

import statistics

eg_list=[2,3,4,5,5,32,13,22]
print(eg_list)

import statistics
print(statistics.mean(eg_list))

import statistics as s
print(s.mean(eg_list))

from statistics import mean as m , median as d
print(m(eg_list))
print(d(eg_list))

x=5
b=10

if x>b:
    print('hi')
else:
    print('ok')
    
a=10
b=20
c=40

if a>b:
    print('a is greater than b')
elif a>c:
    print('a is greater than c')
else:
    print('b is greater than a or c')


def example():
    print('example')
b=20+30
print(b)
example()

def example1():
    print('example1')
    a=10+10
    return a
retun_val=example1();
print(return_val)

def example3(num1,num2):
    print('third function')
    z=num1*num2
    return z
return_value=example3(20,40)
print(return_value)
    

def example4(num1,num2=5):
    print('forth example')
    z=num1**num2
    return z
return_value=example4(4)
print(return_value)


def example5(num1):
    print('fifth example')
    c=num1*example3(30,20)
    return c
print(c)
print(example5(10))


import collections as c

#C:\Users\chandrasen.wadikar\Desktop

f=open('C:\\Users\\chandrasen.wadikar\\Desktop\\test.txt','r')

type(f)

text=f.read()
print(text)
f.close()


f=open('C:\\Users\\chandrasen.wadikar\\Desktop\\test.txt','r')
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
f.close()


f=open('C:\\Users\\chandrasen.wadikar\\Desktop\\test.txt','w')

type(f)

text1=f.write('Say Hi')
print(text1)
f.close()

f=open('C:\\Users\\chandrasen.wadikar\\Desktop\\test.txt','a')
f.write("Hello test again/n")
f.close()

with open('C:\\Users\\chandrasen.wadikar\\Desktop\\test.txt','r') as f:
    read_data=f.read()
    print(read_data)

with open('C:\\Users\\chandrasen.wadikar\\Desktop\\test.txt','a') as f:
    f.write('Final Test')
    print(read_data)
    
import os
pathDir='C:\\Users\\chandrasen.wadikar\\Desktop\\'

f=open(os.path.join(pathDir,'test.txt'),'r')


test=[1,3,4,56,67,4,4,4,2]
for x in test:
    print(x)


test=[2,3,4,5,5,6,6,6,7]
for x in test:
    print(x**2)
    
test=[9,3,5,53,34,22,34,2,1,5,6,5.4,4]
for x in len(test):
    print(x)
    
    
condition=1
while condition<11:
    print(condition)
    condition+=1
    
for x in range(1,11):
    print(x)
    
for x in range(1,11,3):
    print(x)
    
testlist=[3.4,5,3,11,3,4,543,3.3,33]
for x in range(len(testlist)):
        print(testlist[x],type(x))
#       print(testlist[x])
        
testlist1=[3,5,4,4,5,56,4,3,3,3,3,111,13,3]
for x in range(len(testlist1)):
    testlist1[x]=testlist1[x]*2
    print(len(testlist1))
    
for i,x in enumerate(testlist1):
    print(i,':',x,':',x*2)

def d(x):
    return x**2
print(d(8))

g=lambda x:x**2 

print(g(8))

sum=lambda x,y :x+y
sum(1,1)
sum(2,3)

testlist=[3,4,4,231,3,3,4,5,5,12,3,4,7,89,2]
for x in testlist:
    print(g(x))

f=lambda a,b: a if(a>b) else b 
f(100,20)
f(200,22)


import collections as c

eg_list=['red','blue','green','yellow','red']

count=c.Counter(eg_list)
print(count)

list(count.elements())

eg_2=c.Counter(cats=4,dogs=6,puppy=2)
        
print(eg_2)

list(eg_2.elements())

print(c.Counter('abrakababra').most_common(3))

from collections import deque
d=deque('ghi')
for elem in d:
    print(elem.upper())
    
d.append('j')
d.appendleft('f')
d
d.pop()
d.popleft()
list(d)
d[0]
d[1]
d[-1]

list(reversed(d))

'h'in d

d.extend('jkl')
d

d.rotate();d

d.rotate(-1)
d

deque(reversed(d))

d.clear()
d

d.popup()

d.extendleft('abc')
d

d={'banana':3,'orange':2,'apple':1,'mango':4}
OrderedDict
OrderedDict(sorted(d.items(),key=lambda t:t[0]))


import numpy as np

a=np.array([1,2,3])
print(a)

a=np.array([[2,3,4],[5,6,7]])
print(a)

a=np.sort(1).reshape(2,3)
a

a=np.arange(15).reshape(3,5)
a
a.shape

a.ndim

a.dtype.ndim

a.dtype.name

a.itemsize

a.size

a=np.array([1,2,3,4])


b=np.array([(1.2,3,4),(9.2,11,22)])
b
b.dtype.name

np.zeros((3,4))

np.ones((2,3,4),dtype=np.int16)


a=np.arange(6)
a

b=np.arange(12).reshape(4,3)
b

print(np.arange(10000))
print(np.arange(10000).reshape(100,100))

a=np.ones((2,3),dtype=int)
a
b=np.random.random((2,3))

a
a*=3
a

b+=2
b
b+=a
b

a=np.random.random((2,3))
a

a.sum()

b.min()

b.max()

b.std()

b.sum(axis=1)

b.sum(axis=-2)

b.max(axis=1)


b.cumsum(axis=1)
b=np.arange(3)
b

np.exp(b)
np.sqrt(b)

c=np.array([2,3.2,1])

np.add(b,c)

data=np.arange(12).reshape(3,4)
ind=data.argmax(axis=0)
ind

data[1][2]=999
ind=data.argmax(axis=0)
print(ind)


import pandas as pd
import numpy as np

data=np.array([2,3,4,5])
data

s=pd.Series(data)
type(s)
print(s)
print(data)

s=pd.Series(data,index=[100,200,300,400])
s=pd.Series(data)
print(s)


s=pd.Series([1,2,3,4,5],index=('a','b','c','d',''))

data={'a':1,'b':2,'c':3}
s=pd.Series(data)
print(s)

print(s[0])


df.drop('d',axis=1)
print(df[''a'])



print (s[0:4])

print(s['f'])

print(s[['a','b','c','d']])


import pandas as pd
df=pd.DataFrame()
print(df)

data=[['hi',2],['hello',3],['test',5]]
df=pd.DataFrame(data,columns=['word','len'])
print(df)


data=[{'a':10,'b':20,'c':30},{'a':90,'b':100,'c':11}]
df=pd.DataFrame(data,index=['first','second'],columns=['a','b','c'])
print(df)

print(df['a'])

print(df['a'],1)


df['d','e']=[4,88],[5,90]
df['d']=[4,88]
print(df)   


df.drop('d',axis=1)
print(df['a'][0:2])

d={'Name':pd.Series(['Tom','Vijay','Chandrasen','DS']),'Age':pd.Series([24,44,41,22]),'Rating':pd.Series([1,5,1,3])}
df=pd.DataFrame(d)
print(df)

#df.to.csv("C:\\Users\\chandrasen.wadikar\\Desktop\\test.txt",index=False)



def func(x):
    res = 0
    for i in range(x):
        res +=i
        return res
print(func(4))

x=5
y=9

try:
    res=x/y
    print(res)
except:
    print('cannot print')
    
pip install rpy2

import sklearn.datasets as data # to know the available default datasets


def sum(x):
    res = 0
    for i in range(x):
        res +=i
        return res
print(sum(4))

1,3,2,4,5






