#If packages are not working then use below solution##

install.packages('readxl', dependencies=TRUE, repos='http://cran.rstudio.com/')

install.packages("readxl")
install.packages("xlsx")

test<-read.csv("C:/Users/chandrasen.wadikar/Desktop/transactions_data.csv")
test

View(test)

summary(test)

t<-lm(merchant_id ~ user_id+transaction_amount, data= test)
t
summary(t)

plot(t)

hist(t)

