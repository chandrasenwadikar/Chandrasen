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


#LOOPS
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
print(example_function5(10))



#COLLECTION

import collections as c

#COUNTER

