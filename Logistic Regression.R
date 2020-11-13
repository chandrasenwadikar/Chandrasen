install.packages('readxl')
library(readxl)
getwd()

test15<-read.csv("C:/Users/Administrator/Desktop/CreditRisk.csv")
View(test15)

install.packages('dplyr')
library(dplyr)
test15<-mutate(test15,Gender1=ifelse(Gender=='Male',0,1))
test15<-mutate(test15,Married1=ifelse(Married=='Yes',0,1))
test15<-mutate(test15,Education1=ifelse(Education=='Graduate',0,1))
test15<-mutate(test15,Self_Employed1=ifelse(Self_Employed=='Yes',0,1))
test15<-mutate(test15,Property_Area1=ifelse(Property_Area=='Urban',0,1))
test15<-mutate(test15,Loan_Status1=ifelse(Loan_Status=='N',0,1))
View(test15)

factor(test15$Loan_Status1)
test15$Loan_Status1<-factor(test15$Loan_Status1);is.factor(test15$Loan_Status1)
sampletest <-sample(2,nrow(test15),replace=TRUE,prob=c(.8,.2))
Train_smapletest <-test15[sampletest ==1,]
Test_sampletest <-test15[sampletest ==2,]
dim(test15)
test15<-select(test15,4,7,8,4,3,12,16,19)

model_test <-glm(Loan_Status1 ~.,family = binomial, data=Train_smapletest)
dim(Train_smapletest)
View(Train_smapletest)
summary(model_test)

pretest<-predict(model_test,Test_sampletest,type = "response")
pred_actual_df <-mutate(pred_actual_df,pretest=ifelse(pred_val>.5,1,0))

#3aa<-data.frame(pretest,t$st12$ApplicantIncome)

#cc<-data.frame(pretest,test12$Loan_Status1,test12$ApplicantIncome)
plot(model_test)


