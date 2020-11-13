#################predict if a particular flight will be delayed or not ?############

install.packages('readxl')
library(readxl)

flight_data<-read.csv("C:/Users/chandrasen.wadikar/Desktop/Flight1987.csv")
View(flight_data)

dim(flight_data)

install.packages('dplyr')
library(dplyr)

#flight_data<-select(flight_data,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)

flight_data<-mutate(flight_data,Depdelaystatus=(CRSArrTime-CRSDepTime))


flight_data<-select(flight_data,3,4,10,11,12,15,16,17,18)

dim(flight_data)
View(flight_data)

flight_data$Depdelaystatus<-factor(flight_data$Depdelaystatus)
is.factor(flight_data$Depdelaystatus)



dim(flight_data)

flight_sample<-sample(2,nrow(flight_data),replace = TRUE,prob = c(.9,.1))
flight_train<-flight_data[flight_sample==1,]
flight_test<-flight_data[flight_sample==2,]

model_flight<-glm(Depdelaystatus~ .,family = binomial,data = flight_train)
model_flight

pre_flight<-predict(model_flight,flight_test,type="response")

pre_actual_flight<-data.frame(pre_flight,flight_test$Depdelaystatus)

tab1<-table(pre_actual_flight$flight_test.Depdelaystatus,pre_actual_flight$pre_flight)
head(tab1)
#tab3<-table(pre_actual_flight$flight_test.FlightNum,pre_actual_flight$pre_flight)

accuracy<-(sum(diag(tab1))/sum(tab1))



#count=flight_data$FlightNum


############Which weekday/Time of day there is maximum chance that flight will be delayed?##########3

install.packages('readxl')
library(readxl)

data_flight<-read.csv("C:/Users/chandrasen.wadikar/Desktop/Flight1987.csv")
data_flight

View(data_flight)

install.packages('dplyr')
library(dplyr)

data_flight<-select(data_flight,3,4,5,7,10,11,12,13,14,15,16,17)

data_flight$FlightNum<-factor(data_flight$FlightNum)
is.factor(data_flight$FlightNum)


flight_sample<-sample(2,nrow(data_flight),replace = TRUE,prob = c(.9,.1))
flight_train<-data_flight[flight_sample==1,]
flight_test<-data_flight[flight_sample==2,]

model_flight<-glm(FlightNum~ .,family = binomial,data = data_flight)
model_flight

pre_flight<-predict(model_flight,flight_test,type="response")

pre_actual_flight<-data.frame(pre_flight,flight_test$FlightNum,flight_test$DayOfWeek)

tab1<-table(pre_actual_flight$flight_test.FlightNum,pre_actual_flight$pre_flight,pre_actual_flight$flight_test.DayOfWeek)
head(tab1)
#tab3<-table(pre_actual_flight$flight_test.FlightNum,pre_actual_flight$pre_flight)

accuracy<-(sum(diag(tab1))/sum(tab1))


######Create the pie chart/barchart for the delayes and time(day of week)#####

deptime<-c(741,729,749,728,928)
depdelay<-c(11,-1,13,16,5)


pie_res<-pie(deptime,labels = depdelay,main="Status")

sum<-sum(deptime)
test<-round((deptime/sum)*100)
pc<-paste(depdelay,test,'%')

pie_t<-pie(deptime,labels = pc,main = 'update' )

install.packages("plotrix")
library(plotrix)

pie_t<-pie3D(deptime,labels = pc,main = 'update' )
