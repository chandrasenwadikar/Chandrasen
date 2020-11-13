
a <- 1
repeat{  #Repeat Loop
  print(a)
  a=a+1
  if(a>4)
    break
}

c <- 200
repeat{
  print(c)
  c=c+100
  if(c>400)
    break 
}

i <- 1
while(i<10) {  # While loop
  print(i)
  i=i+1
}

t <- 20
while (t<15){
  print(t)
  t=t+1
}

#FOR LOOP

cars <- c("audi","merc","i10")
for (j in cars) {       #J is just variable
  print(j)
} 

fortest <- c("empname","salary")
for (c in fortest){
  print(c)
}

c <-v("population","city,","empname")
for (t in c) {
  print(t)
}

num <-c(1,2,3,4,5,6,11,12)
for(i in num) {
  print(i^2)
}

num <-c(1,2,3,4,5,6,11,12)
for(i in num) {
  print(i+2)
}

#IF Else Condition

x <-4
if (x>0){
  print ("Positive Number")
}else
{
  print ("Negative Number")
}

a <-9
if (a>11) {
  print("+VENumber")
}else
{
  print("-VE number")
}

text <-Chandrasen
if {(text=='CSW')}
print("wrong Text")
}else
{
  print(" Correct Text")
}

#Nested else if

aa <- 9
if(aa < 10){
  print("Inside the if part")
}else if(aa==10){
  print("inside the else if")
}else {
  print ("inside the else part")
}

bb <-20
if (bb >20){
  print ("greater than")
}else if(bb <24){
  print ("over smart")
  
}else {
  print ("Correct value")
}


#General Functions

getwd() #Get the working directory(current direcotry)

list.files() # to see the files in the CWD (current working directory)

Sys.time () #to view current system time

Sys.Date() # to view current system date

attach  # to attach file for default use

attach(emp_sal)
mean(salary)

attach(mtcars)
mode(disp)

detach(emp_sal)

# Reading files in R

read.csv("C:/Users/Administrator/Desktop/HospitalCoststest.csv") # while copying the path from your machine it is with backward slash , you need to change with forward slash and then execute



ctest <-read.csv("C:/Users/Administrator/Desktop/HospitalCoststest.csv")


getwd("HospitalCoststest.csv")

getwd("HospitalCoststest")

getwd()#Using this command you can get your current working directory

#Read Excel

install.packages("readxl")
library(readxl)

ctestxls <- read_excel("C:/Users/Administrator/Desktop/Ecommerce_Dataset_test.xlsx")

cc <- read_excel("C:/Users/Administrator/Desktop/Ecommerce_Dataset_test.xlsx")

#Read Text file

ctextread <-readLines("C:/Users/Administrator/Desktop/test.txt")

tr <- readLines("C:/Users/Administrator/Desktop/testing.txt")



#PLOT -- use for graphs 

xx <-c(2,3,4,5,6,9,11,21,33,50)
plot(xx)

plot(emp_sal $salary)

plot(mtcars $cyl)

plot(city_pop $city,population)

plot(mtcars $cyl,disp)

attach(mtcars)
plot(cyl,disp)

attach(emp_sal)

plot(empname,salary)


plot(mtcars$cyl,mtcars$disp,main= 'Cyl vs Disp',xlab = 'Cylinder',ylab = 'Disposition')


plot (city_pop$city,city_pop$population,main ='City vs Population', xlab= 'City',ylab ='Population')

#Line chart

xx <-c(2,34,56,7,4,5,62,12,34,44,9,8,73)
plot(xx,type='o',main='sample line chart, col="blue")

#histogram

hist(mtcars$disp)

hist (mtcars$disp, breaks=5) # break to controling bin

hist (emp_sal$salary)

hist (city_pop$population)
