getwd()
install.packages('readxl')
library(readxl)

#Credit_Risk<-read.csv("C:/Users/chandrasen.wadikar/Desktop/CreditRisk.csv")
View(Titanic)
Titanic2<-Titanic
View(Titanic2)
install.packages('dplyr')
library(dplyr)

#Credit_Risk <-mutate(Credit_Risk,Loan_status1=ifelse(Loan_Status=='Y',1,0))
#Credit_Risk <-mutate(Credit_Risk,Gender1=ifelse(Gender=='Male',1,0))
#Credit_Risk <-mutate(Credit_Risk,Married1=ifelse(Married=='Yes',1,0))
#Credit_Risk <-mutate(Credit_Risk,Education1=ifelse(Education=='Graduate',1,0))
#Credit_Risk <-mutate(Credit_Risk,Self_Employed1=ifelse(Self_Employed=='Yes',1,0))
#Credit_Risk <-mutate(Credit_Risk,Property_Area1=ifelse(Property_Area=='Urban',1,0))

Titanic2 <-mutate(Titanic2,Age1=ifelse(Age=='Child',1,0))

Titanic2 <-mutate()

Titanic2 <-mutate(Titanic2,Sex1=ifelse(Sex =='Male',1,0))

Titanic <-mutate(Titanic,Sex1=ifelse(Sex=='Male',0,1))
dim(Credit_Risk)
Credit_Risk<-select(Credit_Risk,7,8,9,10,14,15,16,17,18,19)

Credit_Risk$Loan_status1<-factor(Credit_Risk$Loan_status1)

#Credit_Risk$Loan_status1<-factor(Credit_Risk$Loan_staus1);  

#is.factor(Credit_Risk$Loan_status1)

is.factor(Credit_Risk$Loan_status1)


CRS<-sample(2,nrow(Credit_Risk),replace=TRUE,prob = c(.9,.1))
Train_CRS<-Credit_Risk[CRS==1,]
Test_CRS<-Credit_Risk[CRS==2,]

Model_CR1<-glm(Loan_status1~.,family = binomial,data = Train_CRS)
Model_CR1

pred_val <-predict(Model_CR1,Test_CRS,type="response")

pred_actual_df<-data.frame(pred_val,Test_CRS$Loan_status1)

tab<-table(pred_actual_df$Test_CRS.Loan_status1,pred_actual_df$pred_val)

accuracy<-(sum(diag(tab))/sum(tab))
