t.test(mtcars$qsec,mu=17.7,conf.level = .95,alternative = "greater")

t.test(mtcars$qsec,mu=19,conf.level = .95,alternative = "greater")



#In a entrance exam(where score is normally distributed)
#Mean score of test was 76 and the SD was 14
# What is the % of the student scoring 89% or above?

#Probability Distribution

Test_pro <-pnorm(89,mean=76,sd=14,lower.tail = TRUE)
Test_pro <-pnorm(89,mean=76,sd=14,lower.tail = FALSE)

#Linear Regression

speed <-c(4,4,7,7,8,9,10,10,10,11,11,12,12,12,12,13,22,45,21,22)
dist <-c (2,10,22,23,4,5,6,6,2,1,22,43,2,7,8,9,4,5,6,12)
cars <-data.frame(speed,dist)


View(cars)
(plot(cars$speed,cars$dist))
cars1<-cars
cars_model <-lm(cars1$dist ~ cars1$speed)
summary(cars_model)
abline(cars_model)

