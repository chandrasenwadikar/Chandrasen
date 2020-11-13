install.packages('readxl')
library(readxl)

matchknn<-read.csv("C:/Users/chandrasen.wadikar/Desktop/matches.csv")
matchknn

mean(matchknn)
matchdata$season
vect<-c(1,2,3,NA,NaN)
is.na(vect)
is.nan(vect)

View(matchknn)

NonNAindex <- which(!is.na(z))
firstNonNA <- min(NonNAindex)

# set the next 3 observations to NA
is.na(z) <- seq(firstNonNA, length.out=3)


matchknn$season<-factor(matchknn$season)
is.factor(matchknn$season)

matchknn.sample<-sample(2,nrow(matchknn),replace = TRUE, prob = c(.8,.2))
matchknn.train<-matchknn[matchknn.sample==1,]
matchknn.test<-matchknn[matchknn.sample==2,]

matchknn.train1<-matchknn.train[,c(1:3)]
matchknn.test1<-matchknn.test[,c(1:3)]
matchknn.train.lbl<-matchknn.train[,4]

library(class)

matchknn_pred<-knn(train = matchknn.train1,test = matchknn.test1,cl=matchknn.train.lbl,k=8)

