install.packages('readxl')
library(readxl)

Credit_Risk<-read.csv("C:/Users/Administrator/Desktop/CreditRisk.csv")
getwd()
View(Credit_Risk)
install.packages('dplyr')
library(dplyr)

Credit_Risk<-mutate(Credit_Risk,Loan_status1=ifelse(Loan_Status=="Y",1,0))
Credit_Risk<-mutate(Credit_Risk,Education1=ifelse(Education=="Graduate",1,0))
Credit_Risk<-mutate(Credit_Risk,Self_Employed1=ifelse(Self_Employed=="Yes",1,0))

Credit_Risk<-mutate(Credit_Risk,newarea=ifelse(Property_Area=="Urban",2,ifelse(Property_Area=="Rural",0,1)))
Credit_Risk<-mutate(Credit_Risk,Gender1=ifelse(Gender=="Male",1,0))
Credit_Risk<-mutate(Credit_Risk,Married1=ifelse(Married=="Yes",1,0))

dim(Credit_Risk)
View(Credit_Risk)
Credit_Risk<-select(Credit_Risk,1,2,3,4,5,6,7,9)


Credit_Risk$Loan_Status1<-factor(Credit_Risk$Loan_Staus1);is.factor(Credit_Risk$Loan_Staus1)



CRS<-sample(2,nrow(Credit_Risk),replace=TRUE,prob=c(.9,.1))
Train_CRS<-Credit_Risk[CRS==1,]
Test_CRS<-Credit_Risk[CRS==2,]

Model_CR1<-glm(Loan_Status1~.,family = binomial,data=Train_CRS)

pred_val <-predict(Model_CR1,Test_CRS,type="response")
pred_val

pred_actual_df<-data.frame(pred_val,Test_CRS$Loan_Staus1)
pred_actual_df

pred_actual_df<-mutate(pred_actual_df,pred_val=ifelse(pred_val>.5,1,0))
pred_actual_df

tab<-table(pred_actual_df$Test_CRS.Loan_Staus1,pred_actual_df$pred_val)
tab

accuracy<-(sum(diag(tab))/sum(tab))
accuracy

Tpr<-(sum(28/28+0))


#Below function used to calculate combined details of Accuracy and Predicated values

Log_matrix<-function(Tp,Tn,Fp,Fn){
  Tpr <<-(Tp/(Tp+Fn))
  Fpr <<-(Fp/(Fp+Tn))
  Precision <<-(Tp/(Tp+Fp))
  Specificity <<-(Tn)/(Tn+Fp)
  Accuracy <<-(Tp+Tn)/(Tp+Tn+Fp+Fn)
  F1_Score <<-(2*Precision*Tpr)/(Precision+Tpr)
  
}
Log_matrix(295,28,0,15,51,0,25,10,2)


##########################################################

#Decision Tree

install.packages('readxl')
library(readxl)

CTG_1 <-read.csv("C:/Users/Administrator/Desktop/CTG.csv")
View(CTG_1)
dim(CTG_1)

install.packages('rpart')
install.packages('party')
library(rpart)
library(party)#Has the ctree function

CTG_1$NSP<-factor(CTG_1$NSP);               is.factor(CTG_1$NSP)

ctg_dt1 <-ctree(NSP ~ LB+AC+FM,data=CTG_1)

ctg_dt1 <-ctree(NSP ~ LB+AC+FM,data=CTG_1,controls = ctree_control(mincriterion = .90,minsplit = 450))

plot(ctg_dt1)

CTG_sample1 <-sample(2,nrow(CTG_1),replace=TRUE,prob = c(.8,.2))
ctg_train <-CTG_1[CTG_sample1==1,]
ctg_test <-CTG_1[CTG_sample1==2,]

Ctg_dt_predict1 <-predict(ctg_dt1,ctg_tes,type="prob")
ctg_dt_predict <-predict(ctg_dt1,ctg_test,type='response')

pred_actual_df1<-data.frame(ctg_dt_predict,ctg_test$NSP)

install.packages('dplyr')
library(dplyr)
#pred_actual_df1<-mutate(pred_actual_df1,ctg_dt_predict=ifelse(ctg_dt_predict>.5,1,0))

tab1<-table(pred_actual_df1$ctg_test.NSP,pred_actual_df1$ctg_dt_predict)
tab1


accuracy<-(sum(diag(tab))/sum(tab))
accuracy

#######################################################################

###Navie Bayes -- Based on the probability

install.packages("e1071")
library(e1071)

CTG_NB <- navieBayes(NSP ~ LB+AC+FM,data=ctg_train)
ctg_NB_predict<-predict(CTG_NB,ctg_test)
ctg_NB_predict_dataframe<-data.frame(ctg_NB_predict,ctg_test$NSP)
ctg_NB_predict_table <-table(ctg_NB_predict,ctg_test$NSP)
NB_accuracy<-sum(diag(ctg_NB_predict_table)) / sum(ctg_NB_predict_table)*100
NB_accuracy

#Tapply
age<-c(50,32,33,49,53,20)
gender <-c("m","m","f","f","m","m")
bbb<-tapply(age,factor(gender),sum)

aa<-split(iris$Sepal.Length,iris$Species)
sapply(aa,mean)

iris.A<-subset(iris)

##Principle Compoenet Analysis-- works on Eigen Values(staistic term)

##################################

##SVM(Support Vector Machine)
CTG_SVM1 <- (NSP ~ LB+AC+FM,data=ctg_train)
ctg_NB_predict<-predict(CTG_NB,ctg_test)
ctg_NB_predict_dataframe<-data.frame(ctg_NB_predict,ctg_test$NSP)
ctg_NB_predict_table <-table(ctg_NB_predict,ctg_test$NSP)
NB_accuracy<-sum(diag(ctg_NB_predict_table)) / sum(ctg_NB_predict_table)*100
NB_accuracy
