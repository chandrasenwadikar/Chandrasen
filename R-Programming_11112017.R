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
Log_matrix(306,24,21,15,34,5,0,0,4)


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



################################################3

#Random Forest

install.packages('randomForest')

library(randomForest)


install.packages('readxl')
library(readxl)

CTG_1 <-read.csv("C:/Users/Admin/Desktop/CTG.csv")


View(CTG_1)
dim(CTG_1)

#install.packages('rpart')
install.packages('party')
#library(rpart)

#library(party)#Has the ctree function

CTG_1$NSP<-factor(CTG_1$NSP);               is.factor(CTG_1$NSP)

model_ctg_rf<-randomForest(NSP ~ LB+AC+FM,data=CTG_1)

#model_ctg_dt1 <-randomForest(NSP ~ LB+AC+FM,data=CTG_1,controls = ctree_control(mincriterion = .90,minsplit = 450))

plot(model_ctg_rf)

CTG_sample1 <-sample(2,nrow(CTG_1),replace=TRUE,prob = c(.8,.2))
ctg_train <-CTG_1[CTG_sample1==1,]
ctg_test <-CTG_1[CTG_sample1==2,]

Ctg_dt_predict1 <-predict(model_ctg_dt1,ctg_test,type="prob")
ctg_dt_predict <-predict(model_ctg_dt1,ctg_test,type='response')

pred_actual_df1<-data.frame(ctg_dt_predict,ctg_test$NSP)

install.packages('dplyr')
library(dplyr)
#pred_actual_df1<-mutate(pred_actual_df1,ctg_dt_predict=ifelse(ctg_dt_predict>.5,1,0))

tab1<-table(pred_actual_df1$ctg_test.NSP,pred_actual_df1$ctg_dt_predict)
tab1


accuracy<-(sum(diag(tab1))/sum(tab1))
accuracy


#############################################33

#SVM (Support Vector Machine)



#install.packages('randomForest')



install.packages('e1071')
library(e1071)

install.packages('readxl')
library(readxl)

CTG_1 <-read.csv("C:/Users/Admin/Desktop/CTG.csv")


View(CTG_1)
dim(CTG_1)

#install.packages('rpart')
#install.packages('party')
#library(rpart)

#library(party)#Has the ctree function

CTG_1$NSP<-factor(CTG_1$NSP);               is.factor(CTG_1$NSP)

model_ctg_svm<-svm(NSP ~ LB+AC+FM,data=CTG_1)

model_ctg_dt1 <-svm(NSP ~ LB+AC+FM,data=CTG_1,controls = ctree_control(mincriterion = .90,minsplit = 450))

plot(model_ctg_svm)

CTG_sample1 <-sample(2,nrow(CTG_1),replace=TRUE,prob = c(.9,.1))
ctg_train <-CTG_1[CTG_sample1==1,]
ctg_test <-CTG_1[CTG_sample1==2,]



Ctg_dt_predict1 <-predict(model_ctg_svm,ctg_test,type="prob")
ctg_dt_predict <-predict(model_ctg_svm,ctg_test,type='response')

pred_actual_df1<-data.frame(ctg_dt_predict,ctg_test$NSP)

install.packages('dplyr')
library(dplyr)
#pred_actual_df1<-mutate(pred_actual_df1,ctg_dt_predict=ifelse(ctg_dt_predict>.5,1,0))

tab1<-table(pred_actual_df1$ctg_test.NSP,pred_actual_df1$ctg_dt_predict)
tab1


accuracy<-(sum(diag(tab1))/sum(tab1))
accuracy


Log_matrix<-function(Tp,Tn,Fp,Fn){
  Tpr <<-(137/(137+15))
  Fpr <<-(11/(11+11))
  Precision <<-(137/(137+11))
  Specificity <<-(11)/(11+15)
  Accuracy <<-(137+11)/(137+11+11+15)
  F1_Score <<-(2*Precision*Tpr)/(Precision+Tpr)
  
}
Log_matrix(137,11,0,15,11,0,19,4,1)


#########################################################

##########KNN################################

ctg<-read.csv("ctg.csv")

ctg$NSP <-factor(ctg$NSP)

is.factor(ctg$NSP)

ctg_sample<-sample(2,nrow(ctg),replace = TRUE,prob = c(.8,.2))
ctg_train<-ctg[ctg_sample==1,]
ctg_test<-ctg[ctg_sample==2,]

ctg_train1<-ctg_train[,c(1:3)]# To create the data set without the labels(columns to be predicated)
ctg_test1<-ctg_test[,c(1:3)]# To create the data set without the labels(columns to be predicated)
ctg_train_lbl<-ctg_train[,4]#It has only the labels(categories i want to predict)

library(class)

ctg_knn_pred<-knn(train = ctg_train1,test = ctg_test1,cl=ctg_train_lbl,k=80)
ctg_knn_pred

pred_df<-table(ctg_knn_pred,ctg_test$NSP)
pred_df

knn_acc<-(sum(diag(pred_df))/sum(pred_df))*100
knn_acc


accuracy<-(sum(diag(pred_df))/sum(pred_df))
accuracy

###Convert data from standrad normal####
ctg <- mutate(ctg,LB1=(ctg$LB - mean(ctg$LB)/sd(ctg$LB)))
ctg <- mutate(ctg,AC1=(ctg$AC - mean(ctg$AC)/sd(ctg$AC)))
ctg <- mutate(ctg,FM1=(ctg$FM - mean(ctg$FM)/sd(ctg$FM)))
ctg <- mutate(ctg,NSP1=(ctg$NSP - mean(ctg$NSP)/sd(ctg$NSP)))

###################################################################

######CHI-Square##########################3

install.packages('MASS')
library(MASS)

sal<-read.csv("salary_satisfaction.csv")
sal

sal_sat_tab<-table(sal$Service,sal$Salary)
chisq.test(sal_sat_tab)


View(survey)

suerverytab1<-table(survey$Smoke,survey$Exer)
chisq.test(suerverytab1)

#****************************************************

#***********Unsupervised (clustering)- K MEANS*******************

library(dplyr)

ctg_data<-select(ctg, -NSP) # ctg_data<-ctg[,c(1,2,3)] OR ctg_data<-ctg[,c(1:3)]--instated of select this can also work.

model_ctg_kmeans<-kmeans(ctg_data,3,nstart = 1,iter.max = 10)#10 is the number of cluster that want to execute,nstart=1 means only one centroid
model_ctg_kmeans


#Output -(between_SS / total_SS =  82.7 %)-- This output comes basis on the no. of clustering . Behind this Elbow method is used.

View(iris)

iris_data<-select(iris,-Species)# OR iris_data<iris[,c(1,2,3,4)]
head(iris_data)
model_iris_kmeans<-kmeans(iris_data,3,nstart = 1,iter.max = 10)
model_iris_kmeans

View(Credit_Risk)
install.packages('readxl')
library(readxl)
creditrisk1<-read.csv("C:/Users/Admin/Desktop/CreditRisk.csv")
creditrisk1<-na.omit(creditrisk1)
View(creditrisk1)
credit_risk_data<-creditrisk1[,c(4,7,8,9,10)]
head(creditrisk1)
model_cr_kmenas<-kmeans(credit_risk_data,4,nstart = 1,iter.max = 10)
model_cr_kmenas


##########Text Analysis - NLP####################
### Wordcloud analysis ####

install.packages("tm") # Text mining
library(tm)
install.packages("SnowballC") #For stemming
library(SnowballC)
install.packages("stringr") # For split
library(stringr)
install.packages("wordcloud") # For wordcloud
library(wordcloud)



aa<-readLines("C:/Users/Admin/Desktop/modi_speech.txt")
aa
text<-paste(readLines("C:/Users/Admin/Desktop/modi_speech.txt"), collapse = " ") # To make document in proper paragraphs. in first step their is gap , using collpase it is in better format.

print(stopwords())#Find the stop words in document

text2<-removeWords(text,stopwords()) # removing stop words

bag_of_word1 <-str_split(text2," ") #Splits words if there is any space

str(bag_of_word1)

bag_of_word1 <-unlist(bag_of_word1) # to unilist it

str(bag_of_word1) # to check if it is unilisted

wordcloud(bag_of_word1,min.freq = 5,random.order = FALSE) # word cloud is created random.order=FALSE
#prints in desending freq. It is not mandatory to give random.order

#remove words
text2<-removeWords(text2,c("this","poor","let","now"))
text2

length(bag_of_word1)

#### Sentiment Analysis###

install.packages("syuzhet")
library(syuzhet)
install.packages("plyr")
library(plyr)
install.packages("sentimentr")
library(sentimentr)

mysent <-get_nrc_sentiment(text2) # this is inbuild function with the pkg syuzhet
mysent


ab=as.matrix(mysent)

barplot(ab,main ='Modi Speech Sentiment',xlab = 'Sentiment Breakup',ylab = 'Score',col = c('Red'))

#*********************MARKET BASKET******************


install.packages('arules')
install.packages('arulesViz')
library(arules)
library(arulesViz)
library(datasets)

View("Groceries")
str(Groceries)

rules<-apriori(Groceries,parameter = list(supp =.001,conf =.8,maxlen =5))
inspect(rules)

maxlen=5

inspect(rules[1:15])  #inspect top 15 rules

rules1 <-apriori(Groceries,parameter = list(supp = .001,conf =.8),appearance =list(rhs =("whole milk",default ="lhs"))
inspect(rules1)
inspect((rules1[1:15]))

#appearence =list(rhs=c("whole milk","Rice")default ="lhs")

install.packages('readxl')
library(readxl)

cos_ap<-read.csv("C:/Users/Admin/Desktop/Cosmetics.csv")

View(cos_ap)

cos_ap$Foundation<-factor(cos_ap$Foundation)
is.factor(cos_ap$Foundation)

cos_ap_rule<-apriori(cos_ap,parameter = list(supp =.3,conf =.9)),appearance = list(rhs= "Bag=No",default ="lhs"))
inspect(cos_ap_rule)
inspect((cos_ap_rule[1:15]))

rules_cos1<-apriori(cos_ap,parameter = list(supp =.06,conf =.9,minlen =5,maxlen =10),appearance = list(rhs=c("Foundation=Yes"),default ="lhs"))
inspect(rules_cos1)
inspect((rules_cos1[1:10]))
                    
                    
#rules1 <-apriori(Groceries,parameter = list(supp = .001,conf =.8),appearance =list(rhs =("whole milk",default ="lhs"))
  #               inspect(rules1)
   #              inspect((rules1[1:15]))