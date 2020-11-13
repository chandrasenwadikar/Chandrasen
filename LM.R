install.packages('dplyr')
library(dplyr)

testlm<-read.csv("C:/Users/chandrasen.wadikar/Desktop/demandformoney.csv")
testlm

modellm<-lm(Money_printed ~GDP+Interest_RATE+WPI,data = testlm)
summary(modellm)
plot(modellm)