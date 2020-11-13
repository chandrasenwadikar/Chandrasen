###Installing packaging and libraries to read the excel file ####

install.packages('readxl')
library(readxl)

### Define the path of the file and use the vector to read the defined file ###

airline<-read_xlsx("C:/Users/chandrasen.wadikar/Desktop/Airfares1.xlsx")
airline
View(airline)


### Model-1 - Consider all variables and predict the output #####

testmodel_airline<-lm(FARE ~COUPON+NEW+VACATION+SW+HI+S_INCOME+E_INCOME+S_POP+E_POP+SLOT+GATE+DISTANCE+PAX,data = airline)
summary(testmodel_airline)
plot(testmodel_airline)
predict_model<-predict(testmodel_airline,airline)
predict_actual_model<-data.frame(predict_model,airline)


###MODEL-2###

testmodel_airline<-lm(FARE ~COUPON+NEW+VACATION+SW+HI+S_INCOME+E_INCOME+SLOT+DISTANCE+PAX,data = airline)
summary(testmodel_airline)
plot(testmodel_airline)
predict_model<-predict(testmodel_airline,airline)
predict_actual_model<-data.frame(predict_model,airline)


##Model-3

testmodel_airline<-lm(FARE ~COUPON+NEW+VACATION+SW+HI+SLOT+DISTANCE,data = airline)
summary(testmodel_airline)
plot(testmodel_airline)
predict_model<-predict(testmodel_airline,airline)
predict_actual_model<-data.frame(predict_model,airline)


