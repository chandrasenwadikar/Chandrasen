install.packages('readxl')
library(readxl)

flight_data<-read.csv("C:/Users/Chandrasen1/Desktop/Flight1987.csv")
View(flight_data)

dim(flight_data)

install.packages('dplyr')
library(dplyr)

#flight_data<-select(flight_data,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)

flight_data<-mutate(flight_data,Depresult=(CRSDepTime-DepTime))

flight_data<-select(flight_data,3,4,5,6,7,8,10,15,16,17)

dim(flight_data)
View(flight_data)

flight_data$Depresult<-factor(flight_data$Depresult)
is.factor(flight_data$Depresult)



dim(flight_data)

flight_sample<-sample(2,nrow(flight_data),replace = TRUE,prob = c(.9,.1))
flight_train<-flight_data[flight_sample==1,]
flight_test<-flight_data[flight_sample==2,]

model_flight<-glm(FlightNum~ .,Month+DayofMonth+DayOfWeek+DepTime+CRSDepTime+ArrTime+CRSArrTime+UniqueCarrier+ActualElapsedTime+CRSElapsedTime+ArrDelay+DepDelay+Origin+Dest+Distance)
