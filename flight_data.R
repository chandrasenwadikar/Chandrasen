install.packages('readxl')
library(readxl)



flight_data1<-read.csv("C:/Users/chandrasen.wadikar/Desktop/Flight1987.csv")
View(flight_data1)

dim(flight_data1)

install.packages('dplyr')
library(dplyr)

#flight_data<-select(flight_data,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)

flight_data1<-select(flight_data1,3,4,7,8,10,11,12,13,14)

flight_data1$FlightNum<-factor(flight_data1$FlightNum)
is.factor(flight_data1$FlightNum)

dim(flight_data1)

flight_sample<-sample(2,nrow(flight_data1),replace = TRUE,prob = c(.9,.1))
flight_train<-flight_data1[flight_sample==1,]
flight_test<-flight_data1[flight_sample==2,]

model_flight<-glm(FlightNum~.,family = binomial,data = flight_train)
model_flight

predict_test<-predict(model_flight,flight_test,type = "response")

predict_actual_flight<-data.frame(predict_test,flight_test$FlightNum)

tab2<-table(predict_actual_flight$flight_test.FlightNum,predict_actual_flight$predict_test)
tab2

accuracy<-(sum(diag(tab2))/sum(tab2))
#####################################################3333333


flight_data_lm<-read.csv("C:/Users/chandrasen.wadikar/Desktop/Flight1987.csv")
View(flight_data_lm)

flight_test_model<-lm(FlightNum ~Year+Month+DayofMonth+DayOfWeek+DepTime+CRSDepTime+ArrTime+CRSArrTime+UniqueCarrier+ActualElapsedTime+CRSElapsedTime+ArrDelay+DepDelay+Origin+Dest+Distance)
