a <-100
print(a)

x <-21:30
x[5]

class(x)

ve <-21:30
ve2 <-31:40
vec3 <-41:50

result <-matrix(c(ve,ve2,vec3))

class (result)

r1 <- data.frame(ve,ve2,vec3)


sales <-c(100,700,400,66,77,88)
rank <-rank(sales)
sorted <- sort(sales)
ordered <-order(sales)

View <-data.frame(sales,rank,sorted,ordered)


z <-c(6,5,3,7)
which(z*z>9)

z*z>8

letters
which(letters=='w')

paste(a,b,sep = "")

x <-c("apple","tomato","cheery","mango")
grep("a",x)
grep("a",x,value = "TRUE")

word="banana"
chartr(old="b",new="B",word)

substr(word,start=2,stop=5)

word="banana |lime|orange"
v=strsplit(word,split="|",fixed = "TRUE")
v[[1]][1]

x<-matrix(c(4,66,77,99,65,21,999,876,88),nrow = 3)

subs <-matrix(c("English","Maths","Science"),nrow = 3, ncol = 3)

cos(5)
tan(10)

200%%11

b <-c(12,48,99,7)
c <-c(33,2,33,48)
d <-(b+c)
print (d)

aa<- c(99,22,34,99)
matt <-matrix(aa,nrow=2,ncol=3)
matt
matt[1,2]

vector1 <-c(99,88,77) ; vector2 <-c(12,12,33)
result <-array(c(vector1,vector2), dim=c(2,2,2))

install.packages('dplyr')
library(dplyr)

View('mtcars')

View(iris)

empname <-c('a','b','c')
salary <-c(100,200,300)
rating <-c(2,2,1)
empstatus <-data.frame(empname,salary,rating)

summary(empstatus)

str(empstatus)

dim(empname)

View(empstatus)

filter_exam <-filter(empstatus,salary > 200)

select_test<-select(empstatus,empstatus$empname)

empstatus $rating

aa<-mutate(mtcars)

bb<-mutate(empstatus)

bb<-empstatus

cc<-mutate (bb,revised_rating =1,1,1)

summary(empstatus)

arrange_test <-arrange(empstatus,salary)

arrange_test <-arrange(empstatus,desc(rating))

aa <- matrix(data=c(1,2,2,2,5,8,6,7,9), nrow=3,ncol=3,byrow=FALSE)
bb <- matrix(data=c(20,80,190),nrow = 3,ncol = 1)
cc <- solve (aa,bb)


#Loops

a <-10
repeat{
  print(a)
  a=a+1
  if(a>10)
    break
  }
x<-300
repeat {
  print(x)
  x=x-100
  if(x<50)
    break
}

y <-1
while (y<=5){
  print(y)
  y=y+1
}

i <-1
while(i<10){
  print(i)
  i=i+1
}


c <-c("audi","merc","i10")
for(n in c){
  print(n)
}

e <-c("empname","salary","rating")
for(x in e){
  print(x)
}

num <-c(1,2,3,4,5)
for (s in num){
  print(s^2)
}

#If else condition
 
x <-9
if(x>10){
  print("Negative number")
}else
{
  print ("Positive number")
}

text <- Chandrasen
if{ (text == 'Chandrasen')
  print ("Match found")
}else
{
  print("Match not found")
}

a <-100
if(a<102){
  print ("-VE")
}else
{
  print ("+VE")
}

text <-Chandrasen
if {(text=='CSW')}
print("wrong Text")
}else
{
  print(" Correct Text")
}


#Nestated If/else condition

aa <-9
if(a>8){
  print("if")
}else if(aa^2){
  
    print("else if")
}else {
  print("else")
}

bb <-1000
if(bb>1001){
  print("greater than value")
}else if(bb<999){
  print("smaller than value")
}else if(bb==1000){
  print("equal to value")
}else {
  print("hello")
}

#general functions

getwd()
list.files()
Sys.Date()
Sys.time()

attach(empstatus)
mean(salary)
attach(mtcars)
mode(disp)
attach(empstatus)
detach(empstatus)

#Reading files

ctest <-read.csv("C:/Users/chandrasen.wadikar/Desktop/testcsv.csv")

install.packages("readxl")
library(readxl)

c <-read_excel("C:/Users/chandrasen.wadikar/Desktop/Core LOS_5.12.1_PA PLAN_0.2.xls")
print(c)

t <-readLines("C:/Users/chandrasen.wadikar/Desktop/Jira suggestion.txt")
print(t)

#Plot/Graphs

ss <- c(2,3,4,5,10,12,34,9)
plot(ss)
plot (mtcars $cyl)

plot(empstatus $rating)


plot (mtcars $cyl, mtcars $disp, main='Cyl vs Disp',xlab = 'Cylinder', ylab = 'Disposition')

plot (empstatus $empname ,empstatus $salary, main='Employee Status',xlab =' Employee',ylab = 'Status')

#Line graph

c <-c(22,23,4,5,99,28,89,20)
plot(c,type='o',main='Simple Line Chart',col='Red')
#Histogram

hist(mtcars$cyl)

hist(empstatus$rating)

#Pai Chart

GDP <-c(19,29,21,33,34)
country <- c("UK","India","USA","Japan")
pie_test <- pie(GDP,labels = country,main="GDP Description countrywise")

sum_gdp <- sum(GDP)
perc_gdp <-round((GDP /sum_gdp)*100)
perc_count_label <-paste(country,perc_gdp,'%')
GDP_PIE_PERC <-pie(GDP,labels=perc_count_label,main="GDP distribution chart")

l_amt <-c(100,200,300,500,600)
customer <-c ("A","B","C","D","E")
pie_result <-pie(l_amt,labels = customer,main = "Loan details for Customer")

sum <-sum (l_amt)
amt <-round((l_amt /sum)*100)
pc <-paste(customer,amt,'%')
pie_t<-pie(l_amt,labels = pc,main="Loan status")

install.packages("plotrix")
library(plotrix)
pie_t<-pie3D(l_amt,labels = pc,main="Loan status")
GDP_PIE_PERC <-pie3D(GDP,labels=perc_count_label,main="GDP distribution chart")


boxplot(mtcars $disp)
boxplot(empstatus $salary)

emp_status <-as.matrix(empstatus)
heatmap(emp_status,col=heat.colors(256),Rowv= NA,Colv = NA,scale='column')

emp_status <-as.matrix(empstatus)
heatmap(emp_status,Col=heat.colors(256),Rowv = NA,Colv = NA,scale = 'column')

mat_cars <-as.matrix(mtcars)
heatmap(mat_cars,col=heat.colors(256),Rowv = NA,Colv = NA,scale = 'column')

#User defined functions

add_num <-function(a1,a2){
  a3 <-a1*a2
  return(a3)
}
add_num(12,13)

add_char <- function (a){
  text <-a
  return(text)
  
}
add_char("Chandrasen")

bb <-function(a,b,c){
  d <<-factorial(c(a,b,c))
  return(d)
}
bb(1,5,2)

cc_factorial <-function (a1,b1,c1){
  c <<-factorial(c(a1,b1,c1))
  return(c)
}
cc_factorial(11,12,13)

bb_factorial <- function (a1,a2,a3){
  testf<<- factorial(c(a1,a2,a3))
  return(testf)
}
bb_factorial(1,5,3)  



# Train and Test -- Build and test algoritham

irisdata <- iris
irissample1 <- sample(2,nrow(irisdata),replace=TRUE,prob =c(.2,.4))
trainiris1 <- irisdata[irissample1==1,]
testiris1 <- irisdata[irissample1==2,]

empiris <- emp_status
empsample1 <- sample(2,nrow(empiris),replace=TRUE,prob = c(.7,.3))
trainemp <- empiris[empsample1==1,]
testemp <- empiris[empsample1==2,]

#Write files

aa <-c(9,32,44,32)
write.csv(aa,"C:/Users/chandrasen.wadikar/Desktop/resource.csv")

text_test <-("Test the write file")
writeLines(text_test,"C:/Users/chandrasen.wadikar/Desktop/comp.-jira.txt")

#cbind and rbind -- introduce new column and row respectively

aa<-c(12,11,2,3,4,5,6,8);mat_aa<-matrix(aa,2,3);mat_aa
new_col <-c(88,99)
exam <- cbind(mat_aa,(new_col))

vv<-c(87,64,23,44,12,34,455,123,121);mat_vv<-matrix(vv,3,4);mat_vv
new_row<-c(21,33)
res <-rbind(mat_vv,(new_row))


licence()

t.test(mtcars$qsec,mu=17,conf.level = .98,alternative = "greater")
t.test(mtcars$mpg,mu=19,conf.level = .95,alternative="greater")


#In a entrance exam(where score is normally distributed)
#Mean score of test was 76 and the SD was 14
# What is the % of the student scoring 89% or above?

#Probability Distribution

test_pro <-pnorm(89,mean=76,sd=14,lower.tail = TRUE)

test_pro <-pnorm(89,mean=76,sd=14,lower.tail = FALSE)

#Linerar Regression

View(cars)
(plot(cars$speed,cars$dist))
cars1<-cars
cars_model<-lm(cars1$speed ~ cars$dist)
summary(cars_model)
abline(cars_model)
plot(cars_model)
cars1[-23]

###

install.packages("readxl")
library(readxl)
wine <-read.csv("C:/Users/chandrasen.wadikar/Desktop/data_set_wine.csv")

View(data_set_wine)
getwd()
list.files()

view(cars)

View(cars)
cars2<-cars
plot(c(cars$speed,cars$dist))
result_cars<-lm(cars2$speed~cars$dist)
summary(result_cars)
plot(result_cars)
pre_res<-predict(result_cars)
test_pre <-data.frame(pre_res,cars2)


View(state.x77)
x77<-state.x77
colnames(x77)[4]<-"life_exp"
colnames(x77)[6]<-"HS_Grad"
state.x77<-as.matrix(state.x77)

x77<-as.data.frame(x77)
is.data.frame(x77)

testmodel_model <-lm(life_exp ~Population+Income+Illiteracy+Murder+HS_Grad+Frost+Area,data = x77)
summary(testmodel_model)
plot(testmodel_model)
predict_test<-predict(testmodel_model,x77)
predict_actual<-data.frame(predict_test,x77)


################################################
#####LOGISTIC REGRESSION ########
getwd()
install.packages('readxl')
library(readxl)

Credit_Risk<-read.csv("C:/Users/chandrasen.wadikar/Desktop/CreditRisk.csv")
View(Credit_Risk)

install.packages('dplyr')
library(dplyr)

Credit_Risk <-mutate(Credit_Risk,Loan_status1=ifelse(Loan_Status=='Y',1,0))
Credit_Risk <-mutate(Credit_Risk,Gender1=ifelse(Gender=='Male',1,0))
Credit_Risk <-mutate(Credit_Risk,Married1=ifelse(Married=='Yes',1,0))
Credit_Risk <-mutate(Credit_Risk,Education1=ifelse(Education=='Graduate',1,0))
Credit_Risk <-mutate(Credit_Risk,Self_Employed1=ifelse(Self_Employed=='Yes',1,0))
Credit_Risk <-mutate(Credit_Risk,Property_Area1=ifelse(Property_Area=='Urban',1,0))

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


####Below function used to to calculate Ture,false,positive,negative,accuarcy,F1 score etc.

Log_matrix<-function(Tp,Tn,Fp,Fn){
  Tpr <<-(Tp/(Tp+Fn))
  Fpr <<-(Fp/(Fp+Tn))
  Precision <<-(Tp/(Tp+Fp))
  Specificity <<-(Tn)/(Tn+Fp)
  Accuracy <<-(Tp+Tn)/(Tp+Tn+Fp+Fn)
  F1_Score <<-(2*Precision*Tpr)/(Precision+Tpr)
  
}
###########################################################

#############DECISION TREE##############################

install.packages('readxl')
library(readxl)

CTG_1<-read.csv("C:/Users/chandrasen.wadikar/Desktop/CTG.csv")
View(CTG_1)

dim(CTG_1)

install.packages('rpart')
library(rpart)
install.packages('party')
library(party)
install.packages('MASS')
library(MASS)

CTG_1$NSP<-factor(CTG_1$NSP)

is.factor(CTG_1$NSP)

ctg_dt1 <- ctree(NSP~ LB+AC+FM,data = CTG_1)

ctg_dt1

plot(ctg_dt1)

ctg_dt1 <- ctree(NSP~ LB+AC+FM,data = CTG_1,controls = ctree_control(mincriterion = .90,minsplit = 500))

plot(ctg_dt1)

ctg_sample1<-sample(2,nrow(CTG_1),replace = TRUE,prob = c(.9,.2))
ctg_train <-CTG_1[ctg_sample1==1,]
ctg_test <-CTG_1[ctg_sample1==2,]

ctg_predict1<-predict(ctg_dt1,ctg_test,type="prob")
ctg_predict<-predict(ctg_dt1,ctg_test,type="response")

pred_actual_df1<-data.frame(ctg_predict,ctg_test$NSP)

tab1<-table(pred_actual_df1$ctg_test.NSP,pred_actual_df1$ctg_predict)
tab1

accuracy<-sum((diag(tab1)/sum(tab1))

#####################################################################################

########NAVIE BAYES################

install.packages('e1071')

library(e1071)

CTG_NB<-naiveBayes(NSP ~ LB+AC+FM,data=ctg_train)

??navieBayes--- #Not exists in this version
  
################TAPPLY#########

age<-c(30,32,34,40,50,60,70,85)
gender<-c("m","f","f","m","f","f","m","f")
result<-tapply(age,factor(gender),sum)

aa<-split(iris$Sepal.Length,iris$Species)

iris.A<-subset(iris)
#############################33
#Random Forest

install.packages("randomForest")
library(randomForest)

install.packages('readxl')
library(readxl)

CTG_11<-read.csv("C:/Users/chandrasen.wadikar/Desktop/CTG.csv")
View(CTG_11)

install.packages('rpart')
library(rpart)
install.packages('party')
library(party)

CTG_11$NSP<-factor(CTG_11$NSP)
is.factor(CTG_11$NSP)

model_ctg_rf<-randomForest(NSP~ LB+AC+FM,data = CTG_11)
model_ctg_rf

plot(model_ctg_rf)

model_ctg_rf<-randomForest(NSP~ LB+AC+FM,data = CTG_11,controls= ctree_control(mincriterion = .90,minsplit = 500))

CTG_sample_rf<-sample(2,nrow(CTG_11),replace = TRUE,prob = c(.9,.1))
CTG_train_rf<-CTG_11[CTG_sample_rf==1,]
CTG_test_rf<-CTG_11[CTG_sample_rf==2,]

CTG_predict_rf<-predict(model_ctg_rf,CTG_test_rf,type="prob")
CTG_predict_rf_1<-predict(model_ctg_rf,CTG_test_rf,type='response')

CTG_actual_prob_rf<-data.frame(CTG_predict_rf,CTG_test_rf$NSP)
CTG_actual_prb_rf_1<-data.frame(CTG_predict_rf_1,CTG_test_rf$NSP)


tab11<-table(CTG_actual_prb_rf_1$CTG_test_rf.NSP,CTG_actual_prb_rf_1$CTG_predict_rf_1)

accuracy<-sum((diag(tab11)/sum))


########################################################################################


####Support Vector Machine (SVM)  ########


install.packages('readxl')
library(readxl)

CTG_SVM<-read.csv("C:/Users/chandrasen.wadikar/Desktop/CTG.csv")
View(CTG_SVM)

install.packages('rpart')
library(rpart)
install.packages('party')
library(party)
install.packages('e1071')
library(e1071)

CTG_SVM$NSP<-factor(CTG_SVM$NSP)
is.factor(CTG_SVM$NSP)

model_ctg_svm<-svm(NSP~ LB+AC+FM,data=CTG_SVM)

model_ctg_svm



model_ctg_svm<-svm(NSP~ LB+AC+FM,data=CTG_SVM,controls=ctree_control(mincriterion = .90,minsplit = 600))

plot(model_ctg_svm)

svm_sample<-sample(2,nrow(CTG_SVM),replace = TRUE,prob = c(.9,.1))
svm_train<-CTG_SVM[svm_sample==1,]
svm_test<-CTG_SVM[svm_sample==2,]

svm_predict<-predict(model_ctg_svm,svm_test,type="response")
svm_predict1<-predict(model_ctg_svm,svm_test,type="prob")

pred_actual_svm<-data.frame(svm_predict,svm_test$NSP)
pred_actual_svm1<-data.frame(svm_predict1,svm_test$NSP)

tab11<-table(pred_actual_svm1$svm_test.NSP,pred_actual_svm1$svm_predict)

accuracy<-sum((diag(tab12)/sum(tab12))


###########################KNN(K NEAREST NEIGHBOUR)#################

install.packages('readxl')
library(readxl)

ctg<-read.csv("C:/Users/chandrasen.wadikar/Desktop/CTG.csv")

View(ctg)

ctg$NSP<-factor(ctg$NSP)
is.factor(ctg$NSP)

ctg.sample<-sample(2,nrow(ctg),replace = TRUE,prob = c(.8,.2))
ctg_train123<-ctg[ctg.sample==1,]
ctg_test111<-ctg[ctg.sample==2,]

ctg_train1<-ctg_train123[,c(1:3)]
ctg_test1<-ctg_test111[,c(1:3)]
ctg_train_lbl<-ctg_train123[,4]

library(class)

ctg_knn_pred<-knn(train = ctg_train1,test = ctg_test1,cl=ctg_train_lbl,k=80 )
ctg_knn_pred

pred_df<-table(ctg_knn_pred,ctg_test111$NSP)

knn_acc<-sum((diag(pred_df))/sum(pred_df))*100
knn_acc

####Convert data from std. normal#####

install.packages('dplyr')
library(dplyr)


ctg <- mutate(ctg,LB1=(ctg$LB - mean(ctg$LB)/sd(ctg$LB)))
ctg <- mutate(ctg,AC1=(ctg$AC - mean(ctg$AC)/sd(ctg$AC)))
ctg <- mutate(ctg,FM1=(ctg$FM - mean(ctg$FM)/sd(ctg$FM)))
ctg <- mutate(ctg,NSP1=(ctg$NSP - mean(ctg$NSP)/sd(ctg$NSP)))


##############CHI SQUARE##################################


install.packages('readxl')
library(readxl)

sal<-read.csv("C:/Users/chandrasen.wadikar/Desktop/salary_satisfaction.csv")
View(sal)

install.packages('MASS')
library(MASS)

sal_sat_tab<-table(sal$Service,sal$Salary)
sal_sat_tab

chisq.test(sal_sat_tab)

View(survey)

survery_test<-table(survey$Smoke,survey$Exer)
chisq.test(survery_test)


##########UNSUPERVISED ALOGORITHM########################3

###################K-MEANS############################

install.packages("dplyr")
library(dplyr)

install.packages('readxl')
library(readxl)

ctg_data<-read.csv("C:/Users/chandrasen.wadikar/Desktop/CTG.csv")

ctg_data<-select(ctg,-NSP)

View(ctg_data)

#model_ctg_kmeans<-kmeans(ctg_data,3,nstart =1,iter.max=10)


model_ctg_k<-kmeans(ctg_data,3,nstart = 1,iter.max = 10)


View(iris)

head(iris_data)

iris_data<-select(iris,-Species)

model_iris_kmeans<-kmeans(iris_data,3,nstart = 1,iter.max = 10)
model_iris_kmeans

View(Credit_Risk)

install.packages('readxl')
library(readxl)


creditrisk1<-read.csv("C:/Users/chandrasen.wadikar/Desktop/CreditRisk.csv")
creditrisk1<-na.omit(creditrisk1)

View(creditrisk1)

credit_risk_data<-creditrisk1[,c(4,7,8,9,10)]

model_cr_kmeans<-kmeans(credit_risk_data,4,nstart = 1,iter.max = 5)
model_cr_kmeans


#############TEXT ANALYSIS##########################

###Wordcloud analysis

install.packages("tm")#Text mining 
library(tm)
install.packages("SnowballC")#For stemming
library(SnowballC)
install.packages("stringr")#For split
library(stringr)
install.packages("wordcloud")#For wordcloud
library(wordcloud)

aa<-readLines("C:/Users/chandrasen.wadikar/Desktop/modi_speech.txt")
aa

text<-paste(readLines("C:/Users/chandrasen.wadikar/Desktop/modi_speech.txt"),collapse = " ")

print(stopwords())#Find the stop words in document

text2<-removeWords(text,stopwords()) #removing stop words

bag_of_word1<-str_split(text2," ")#Split the words , if there is any space

str(bag_of_word1)

bag_of_word1<-unlist(bag_of_word1) #to unlist it

str(bag_of_word1) #to check if it is unlist

wordcloud(bag_of_word1,min.freq = 5,random.order = FALSE) # Wordcloud created

#Remove Words

text2<-removeWords(text2,c("this","poor","let","now"))

length(bag_of_word1)

#Sentiment Analysis

install.packages("syuzhet")
library(syuzhet)
install.packages("plyr")
library(plyr)
install.packages("sentimentr")
library(sentimentr)


mysent<-get_nrc_sentiment(text2)
mysent

ab=as.matrix(mysent)

barplot(ab,main = 'Modi Speech Sentiment',xlab = 'Sentiment Breakup',ylab = 'Score',col = c('Orange'))

##################MARKET BASKET ANALAYSIS#############################

install.packages("arules")
library(arules)
install.packages("arulesViz")
library(arulesViz)
library(datasets)

install.packages('readxl')

library(readxl)

data()


groceries_data<-read.csv("C:/Users/chandrasen.wadikar/Desktop/groceries.csv")

View(groceries_data)


str(groceries_data)

rules<-apriori(groceries_data,parameter = list(supp =.001,conf =.8,maxlen =5))
inspect(rules)
inspect(rules[1:15])

rules1 <-apriori(groceries_data,parameter = list(supp =.003,conf =.6))



cos_ap<-read.csv("C:/Users/chandrasen.wadikar/Desktop/Cosmetics.csv")
View(cos_ap)

cos_ap$Foundation<-factor(cos_ap$Foundation)
is.factor(cos_ap$Foundation)

cos_ap_rule<-apriori(cos_ap,parameter = list(supp =.3,conf =.9))
inspect(cos_ap_rule)
inspect(cos_ap_rule[1:15])

rules2<-apriori(cos_ap,parameter = list(supp =.3,conf =.9,minlen =5,maxlen =10),appearance = list(rhs=c("Foundation=Yes",default ="lhs"))
inspect(rules2)   
inspect(rules2[1:15])

#######################TIME SERIES###############################

View(AirPassengers)
plot(AirPassengers)
summary(AirPassengers)
cycle(AirPassengers)
decompose(AirPassengers)

#Differenting AR and MA

d1_AP1<-diff(AirPassengers)
plot(d1_AP1)
d1_AP2<-diff(d1_AP1)
plot(d1_AP2)
d1_AP3<-diff(d1_AP2)
plot(d1_AP3)

diff(diff(AirPassengers))

acf(AirPassengers)
pacf(AirPassengers)

model_AP1<-arima(AirPassengers,c(1,1,1))
model_AP2<-arima(AirPassengers,c(1,2,4))
model_AP3<-arima(AirPassengers,c(2,2,0))
model_AP4<-arima(AirPassengers,c(2,1,1))
model_AP5<-arima(AirPassengers,c(1,2,2))
model_AP6<-arima(AirPassengers,c(4,2,4))

model_air_predict<-predict(model_AP6,n.ahead =5) #n.ahead = 5 means how many iteration we want to predict, here it is 5 times

##############DATE FUNCTIONS#################

install.packages('lubridate')
library(lubridate)

week(ymd("2020-12-11")) # Shows the week of the year
day(ymd("2022-12-11"))# shows the dat of the month
month(ymd("2022-11-10")) # Shows the month of the year

date1<-ymd("2015-01-01")
date2<-ymd("2018-06-06")

difftime(date1,date2,units = 'days')
difftime(date1,date2,units = 'hour')
difftime(date1,date2,units = 'months')

d1<-as.Date("2015-12-12")
weekdays(d1)

date3<-ymd("2015-01-01")
aa<-seq(date1,date2,by ='week')
bb<-seq(date1,date2,by ='2 week')

iqtest<-c(10,22,20,99,234,56,37,89)

z.test(iqtest,100,89)

??z.test

View(Titanic)

data()
