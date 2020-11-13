#PIE Chart

GDP <-c(19.4,9,23.8,33.9,3,5,6,7.9)
mean(GDP)
countries <-c("UK","India","Japan","Korea","USA")
GDP_PIE <-pie(GDP,labels = countries,main="gdp distribution")

sum_gdp <- sum(GDP)

perc_gdp <-round((GDP /sum_gdp)*100)

perc_count_label <-paste(countries,perc_gdp,'%')

GDP_PIE_PER<-pie(GDP,labels=perc_count_label,main="GDP distribution in % of countires")


empdetails <-c("Name","salary")
rating <-c(1,2,3,4,9)
result <-pie3D(empdetails,lables=rating,main="result123")

install.packages("plotrix")
library (plotrix)

GDP_PIE_PER<-pie3D(GDP,labels=perc_count_label,main="GDP distribution in % of countires")


#Box Plot  shows the minimum,maximum,median,1st quartile and 3rd quartile 

boxplot(mtcars $disp,main='Cylinder Summary')

boxplot(emp_sal $salary, main ='Details')

#HeatMap

matt_cars <-trix(mtcars)
heatmap(matt_cars,col=heat.colors(256),Rowv=NA,Colv=NA,scale='column')

#User defined functions

add_num <- function  (a1,a2){
  a3 <-a1*a2  # a3 is local variable here , because it is defined inside the body
  return (a3)
}
add_num(12,10)


aa_num <- function (a1,a2){
  a3 <<- a1+a2 # to access the value outside the function use << -
  ab = a1-a2
  print(a3)
  return(a3)
}

aa_num(2,23)

bb_num <- function (a1,a2,a3,a4){
  a5 <<- a1+a2+a3+a4
  return(a5)
}
bb_num(10,20,13,29)

mode_test <- function(m1,m2,m3,m4){
  m2 <<- mode(c(m1,m2,m3,m4))
  return(m2)
}
mode_test(90,23,34,88)

mode_test <- function(m1,m2,m3,m4){
  m2 <<- mean(c(m1+m2+m3+m4)/4)
  return(m2)
}
mode_test(90,23,34,88)


bb_factorial <- function (a1,a2,a3){
  testf<<- factorial(c(a1,a2,a3))
  return(testf)
}
bb_factorial(1,5,3)  


#Train and Test for Build and test algorithm

irisdata <- iris #Sample remains the same when using the set.seed function
irissample1 <- sample(2,nrow(irisdata),replace=TRUE,prob = c(.7,.3)) #prob =probability
trainiris1 <- irisdata[irissample1==1,]# after 1 comma --indicates to select all data
testiris1 <- irisdata[irissample1==2,]# after 2 comma --indicates to select all data


#Write Function
aa <-c (1,2,3,4,5)
write.csv(aa,"C:/Users/Administrator/Desktop/Chandrasen/HospitalCoststest.csv")

xamp_text_file <-('this will be created as a text file for test')
#writeLines(xamp_text_file,"path/testing.txt")
writeLines(xamp_text_file,"C:/Users/Administrator/Desktop/Chandrasen/testing.txt")

#Cbind and rbind -- c bind used for add your columns and r bind is for add rows

aa <-c(9,4,5,6,10,11);mat_aa <-matrix(aa,2,3);mat_aa #create matrix
new_col <-c(77,88)
exm_bind <-cbind(mat_aa,(new_col));

aa <-c(9,4,5,6,10,11);mat_aa <-matrix(aa,2,3);mat_aa #create matrix
new_row <-c(77,88)
exm_bind <-rbind(mat_aa,(new_row));
















t.test(mtcars$qsec,mu=17.7,conf.level = .95,alternative = "greater")

t.test(mtcars$qsec,mu=19,conf.level = .95,alternative = "greater")



#In a entrance exam(where score is normally distributed)
#Mean score of test was 76 and the SD was 14
# What is the % of the student scoring 89% or above?

#Probability Distribution

Test_pro <-pnorm(89,mean=76,sd=14,lower.tail = TRUE)
Test_pro <-pnorm(89,mean=76,sd=14,lower.tail = FALSE)

#Linear Regression

speed <-c(4,4,7,7,8,9,10,10,10,11,11,12,12,12,12,13,22,45,21,22,30,22,60,70,95,120)
dist <-c (2,10,22,23,4,5,6,6,2,1,22,43,2,7,8,9,4,5,6,12,50,30,44,10,5,2)
cars <-data.frame(speed,dist)


View(cars)
(plot(cars$speed,cars$dist))
cars1<-cars
cars_model <-lm(cars1$dist ~ cars1$speed)
summary(cars_model)
predict_cars <- predict(cars_model,cars1)
predict_cars
aa <-data.frame(predict_cars,cars1$dist,cars1$speed)
print(aa)
abline(cars_model)

plot(cars_model)
cars1[c(-21,-23),]


error <-(predict_cars-cars$dist)#predict minus actual
errorsq = error^2 # do the square
errorsq_sum <-sum(errorsq)
DoF <- 24 # Degree of freedom , and 24 specific for this example.
Res_Std_Err <- sqrt(errorsq_sum/DoF)
Res_Std_Err

#
#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)# Decide the Hypothesis and can consider based on the  below (***)  
#(Intercept)  -1.7465(Y Axis)     8.2963  -0.211   0.8375  
#cars1$speed   2.1549(X Axis)     0.9271   2.324   0.0425 * # If getting *** is the sign to play important role in predication 
  ---
 # Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 8.131 on 10 degrees of freedom
#Multiple R-squared:  0.3508,	Adjusted R-squared:  0.2859 
#F-statistic: 5.403 on 1 and 10 DF,  p-value: 0.04245   # analysis of Variance i.e. ANNOVA

View(mtcars)
bb<-lm(mpg~.,data = mtcars)
aa<-predict(bb,mtcars)
cc <-data.frame(aa,mtcars$disp,mtcars$cyl)
summary(mtcars)
abline(mtcars)

plot(mtcars)


mtcarsnew_model1 <-lm(mpg ~ disp+hp+wt+qsec+am,data=mtcars)
plot(mtcarsnew_model1)
summary(mtcarsnew_model1)
predict(mtcarsnew_model1)

mtcarsnew_model2 <-lm(mpg ~disp+hp+wt+am+qsec+cyl+gear+carb,data = mtcars)
plot(mtcarsnew_model2)
summary(mtcarsnew_model2)
predict(mtcarsnew_model2)

View(state.x77)
x77 <-state.x77
colnames(x77)[4]<-"life_exp"
colnames(x77)[6]<-"HS_Grad"
state.x77 <-as.matrix(state.x77)

x77<-as.data.frame(x77)
is.data.frame(x77)

statetest_model <-lm(life_exp ~Population+Murder+HS_Grad+Frost,data=x77)
summary(statetest_model)
plot(statetest_model)
predict_statetest<-predict(statetest_model,x77)
predict_actual_statetest <-data.frame(predict_statetest,x77)

install.packages('readxl')
library(readxl)
read12 <-read.csv('C:/User/Lenovo6/Desktop/LungCapData.csv')
getwd()
read.csv('LungCapData.csv')

install.packages('dplyr')
library(dplyr)

LungCap <-mutate(LungCap,Smoke1=ifelse(Smoke=="no",0,1))#change the no and yes in 0 and 1
LungCap <-mutate(LungCap,Gender1=ifelse(Gender=="female",0,1))
LungCap<-mutate(LungCap,Caesarean1=ifelse(Caesarean=="no",0,1))
LungCap_model <-lm(LungCap ~Age+Height+Smoke1+Gender1+Caesarean1,data=LungCap)
summary(LungCap_model)
plot(LungCap_model)

LungCapdata<-LungCap
LungCapsample1 <-sample(2,nrow(LungCap),replace = TRUE,prob = c(.9,.1))
trainlungs1 <-LungCap[LungCapsample1==1,]
testlungs1 <-LungCap[LungCapsample1==2,]

model_LCN <-lm(LungCap ~.,data=trainlungs1) #Consider Train in data model.

LungCap_model <-lm(LungCap ~Age+Height+Smoke1+Gender1,data=LungCap)
summary(LungCap_model)
Lungpredict<-predict(LungCap_model)
aa<-data.frame(Lungpredict,LungCap$Height,LungCap$Gender,LungCap$Age)

first_ten <-mtcars[1:10] # select first 10 elements
aa <- mtcars [c(1,6)]#To select 1st and 6th row and all
bb <- mtcars[,c(1,3,5)]#all rows and only 1st,3rd and 5th column
bb<-mtcars[c(1:10)]#1st 10 rows and all columns
cc <-mtcars[c(1:5,8),]#1st 5th rows and 8th row and all column
bb <-iris [c(2:10,16,19,150),c(1,4)]
