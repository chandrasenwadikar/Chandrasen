#swirl- self learning environment

#Operater

8+5 #addition
8-5 #substraction
9*2 #multiplication
9/2 # division

2+3/5+5

(2+3)/(5+5)

9 %%2 #reminder

3^5 #exponent or power operater

#Relation operators- output woudl be always in boolean (True/False)

8>7

8>=8

8<7

8==8

8<=7

# Functions

sqrt(2)

exp(3)

sin(10)

cos(1.572)

#Help

help(log)
?log

log10(10)

abs(-5.6)
help(abs)
?abs

#Complex expression

sin(log10(sqrt(9-8)))


# Varirables

x <-5
y<-6
y
x<=y
cars<-3
Scotter<-4
cars
Scotter

"my"+"apples"



class("my")
class(1111)

"Fe" == "fe" 
x=2;y=T;z="apple"
typeof(x);typeof(y);typeof(z)

#sequences

1:5
20:45

#Vectors 

age<-c(20,30,40,55,60,62,84)
age

number<-c(3,8,9,10)
number

y<-c(1.4,"ten")
y
#Coercion - Means vectors will identify most corercion (i.e most important data types which are commnly used)
class(c(FALSE,3,"Twleve")) #Her boolean,number and charaters are defined , but coercion identify for this record as charater. Because charater in genric in nature
class(c(FALSE,3)) #Like above nummeric in genric in nature.
class(c(FALSE)) #logical
#accessing vectors 
my_vector1<- c(3,8,9,10)
my_vector
my_vector[3] # retruns in thrid element. Square bracket indicates subscript/indexing
my_vector[2:4]

v1 = c(3, 4, 5, 6, 7, 8)
v1
# Which are the following options consider to access 4th element from above vector
v1(1:4)
v1[1,2,3,4]
v1[1:4]#correct answer
v1[3,4,5,6]
my_vector[-3]
length(my_vector)

my_vector1<- c(3,8,9,10)
my_vector2<- c(3,8,9,10)

add<-my_vector1+my_vector2
add


a = c(2 ,6)
b = c(4, 3, 5, 7) 

d=a+b
d

my_vector1<- c(3,8,9,10)
min(my_vector1)
sort(my_vector1)


my_vector1<- c(3,8,9,NA,10,NA)
is.na(my_vector1)# is.na used to identify the NA elements in the defined list.

a = c(2 ,6, 3) ; b = c(4, 3, 5, 7) 


#install SWIRL

library(swirl)
swirl()
Chandrasen

#Factors

dice  <-  c(1, 2, 4, 5, 5, 3, 2, 6, 3, 5, 6, 2, 1, 4, 3, 6, 5, 3, 2, 2, 5) 
dice_levels<-factor(dice)
dice_levels

heights<-c("medium","shor","tall","medium","medium")
heights
heights[1]<-"ultra-short"
heights

factor_heights<-factor(heights)
factor_heiights
levels(factor_heights)
summary(factor_heights)

#Matrices

matrix(1:9, byrow = TRUE, nrow = 3)

matrix(1:9,byrow = FALSE,nrow = 3)

#multiple vecotrs combining in matrix
karan_dice<-c(3,1,5,7,6,1)
raj_dice<-c(6,1,3,4,6,1)
ajay_dice<-c(2,1,4,2,2,5)

vector_dice<-c(karan_dice,raj_dice,ajay_dice)
vector_dice
dice_matrix<-matrix(vector_dice,nrow = 3,byrow = TRUE)
dice_matrix

#exisitng in data sets in R

iris
head(iris)
tail(iris)
str(iris)


#Data Frames

player_name<-c("Rohit","Kohli","dhoni")
total_runs<-c(5000,7200,8900)
strike_rate<-c("82.22","89.12","87")

team<-data.frame(player_name,total_runs,strike_rate)
team
str(team)

#strings as factors

team_char<-data.frame(player_name,total_runs,strike_rate,stringsAsFactors = FALSE)
team_char

#accessing elelements
team

team[2,1]

team[2,]
team[,2]
team[,"total_runs"]
team$total_runs

team[2:3,1:2]
team[1:3,"total_runs"]

team[1,3]
team[,(1,3)]

#perform operations

player_name<-c("jadega","ashwin","raina")
total_runs<-c(1200,2200,5000)
strike_rate<-c("34","45","80")

team2 <-data.frame(player_name,total_runs,strike_rate)
team2

#combining two data frames
complete_team<-rbind(team,team2)
complete_team

player_age<-c(29,26,230)
player_age

hit_six<-c(230,123,133)
hit_six

team3<-data.frame(player_age,hit_six)
team3

team1_info<-cbind(team,team3)
team1_info

team1_info<cbind(team,player_age,hit_six)
team1_info

nrow(team1_info)

team1_info
ncol(team1_info)

summary(team1_info) #returns summary of attributes

sum(complete_team$total_runs)

#Invoking data set from external sources

team_from_text_file<-read.table("myfile.txt")
team_from_text_file

team_from_csv<-read.csv("myfile.csv")
team_from_csv

team_from_csv$hit_six

#Lists

Stadiums <- c("wankhede","Eden garden","kolta")
per_matrix<-matrix(1:6,nrow = 2)
per_matrix

mylist<-list(team, Stadiums,per_matrix)
mylist
mylist[[2]]
mylist[["ages"]]


mylist[["Stadiums"]]

mylist[["St"]]

mylist[2,]

mylist[[2]][2]

mylist[1][2, ]

mylist[1][ ,2]

mylist[[1]][2, ]


getwd()

##SESSION 2 ####

bank <-read.csv("bank.csv",stringsAsFactors = F)
bank

str(bank)

bank$marital<-factor(bank$marital)
str(bank$marital)

subset1<-data.frame(bank$age,bank$salary,bank$y)
str(subset1)

bank[1,]

#Logical and Relational Operators

8 >c(2,9,6,8,10)
c(8,5,7,1,0) >c(2,9,6,8,0)
c(8,5,7,1,0)>c(2,9,6,8)

#Logical operators
#And (&)
6>5 & 7>4
6>5 & 1>4

#OR operator

6>5 | 7>4
6>5 | 1>4

#NOT Operator

!TRUE

7 !=6
7!=7
c(8,5,7,1,0) !=c(2,9,6,8,0)


 c(7, 3, 2, 8, 0, 4)> c(2, 1, 3, 4, 2, 4)
 
 c(2, 1, 3, 4, 2, 4) >c(7, 3, 2, 8, 0, 4)
 
 #Conditional Statement 
 
 shopping_bill<-c(90,130,52,75,70,24,72,125,90,68,56,50,85)
 total<-sum(shopping_bill)
 total

 if (total>1000)
 {
   print("You are out of cash !")
 }else if (total<900){
   print("Go with shopping")
 }else{
   print("Only shopping , and not chocolate !")
 }
 
 person <- bank[1,]
 person
 
 if (person$marital=="married")
 {
   if(person$housing=="yes"| person$salary >60000)
   {
     print("Issue credit card")
   }
   else
   {
     print("Sorry, not eligible for credit card")
   }
}else if (person$marital=="single")
{
  if(person$education =="tertiary" & person$salary > 40000)
  {
    print("Issue credit card")
  }
  else
  {
    print("Sorry not eligible for credit card")
  }
}

 #Loops
 
 print("Hello")
 print("Hello")
 print("Hello")
 print("Hello")
 print("Hello")
 
 for(i in 1:5)
 {
   print("Hello")
 }
 
 #Applying for bank case study
 
 person1 <-bank[1,]
person2 <-bank[2,]

for(i in 1:nrow(bank))
{
  person <-bank[i,]
  person
  
  if(person$marital=="married")
  {
    if(person$housing=="yes"| (!is.na(person$salary)&person$salary)>60000)
    {
      bank[i,"my_decision"] <-"yes"
    }
    else
    {
      bank[i,"my_decision"] <- "no"
    }
  }else if(person$marital=="single")
  {
    if(person$education=="tertiary" & !is.na(person$salary)& person$salary >40000)
    {
      bank[i,"my_decision"] <- "yes"
    }
    else
    {
      bank[i,"my_decision"] <-"no"
    }
  }
}

#assignment

A_upvotes <- c(7, 3, 2, 8, 0, 4)

if(mean(A_upvotes) >= 4)
{
  print("Congratulations, you won the Popular Badge")
} else if(mean(A_upvotes) >= 3)
{
  print("You are quite close to winning a Badge. Keep working hard.")
}

###
if(mean(A_upvotes) >= 8)
{
  print("Congratulations, you won the Superstar Badge")
} else if(mean(A_upvotes) >= 4)
{
  print("Congratulations, you won the Popular Badge")
} else 
{
  print("You are quite close to winning a Badge. Keep working hard.")
}


#Functions

mean(bank$age)
sd(bank$age)

sum(is.na(bank))
is.na(bank)

sum(is.na(bank$salary))

which(is.na(bank$salary))

mean(bank$age,na.rm = TRUE) #rm = remove
sd(bank$age,na.rm = TRUE)

?mean

max(bank$age)

which.max(bank$age)

bank[which.max(bank$age),]

length(which(bank$y=="yes"))

length(which(bank$my_decision=="yes"))

length(which(bank$y=="yes" & bank$marital=="single"))
#assignments
A_upvotes <- c(7, 3, 2, 8, 0, 4)

which(sum(A_upvotes >= 4))

which(length(A_upvotes >= 4))

which(A_upvotes >= 4)

length(A_upvotes >= 4)
length(which(A_upvotes >= 4))
sum(which(A_upvotes >= 4))
mean(which(A_upvotes >= 4))

#Creating your own functions

credit_card_decision <- function(p)
{
  if(p$marital =="married")
  {
    if(p$housing =="yes" |(!is.na(p$salary) & p$salary)>60000)
    {
      decision <-"yes"
    }
    else
    {
      decision <-"no"
    }
    
    }else if(p$marital =="single")
    {
      if(p$education == "tertiary"& !is.na(p$salary)& p$salary >40000)
      {
        decision <-"yes"
      }
      else
      {
        decision <-"no"
      }
      
      }else
      {
        decision="no"
      }
  return(decision)
}

for(i in 1:nrow(bank))
{
  person<-bank[i,]
  bank[i,"my_decision"] <-credit_card_decision(person)
}


function_math <- function(x, y){
  z <- x + y
  p <- x * y
  q <- z / p
  return(c(z, p, q))
}
function_math(2, 3)


fun_square<- function(vector_in){
  vector_out <- vector_in^2
  return(vector_out)
}

#Apply Family Functions
v

bank <-read.csv("bank.csv",stringsAsFactors = F)
bank

for (i in 1:ncol(bank)) {
  bank[,i] <-factor(bank[,i])}

str(bank)

bank2<-data.frame(sapply(bank, factor))
str(bank2)

marks<-c(23,12,15,30,40)
sapply(marks,function(x) x/40*100 )

#assignments

A_upvotes <- c(7, 3, 2, 8, 0, 4)
B_upvotes <- c(2, 1, 3, 4, 2, 4)
C_upvotes <- c(3, 2, 3, 4, 6, 2)
D_upvotes <- c(4, 2, 5, 3, 3, 1)

combined_vector <- c(A_upvotes, B_upvotes, C_upvotes, D_upvotes)
combined_df <- data.frame(A_upvotes, B_upvotes, C_upvotes, D_upvotes)


sapply(mean, combined_vector)
sapply(combined_vector, mean)
sapply(mean, combined_df)
sapply(combined_df, mean)

v <- sapply(combined_df, max)
v[3]
