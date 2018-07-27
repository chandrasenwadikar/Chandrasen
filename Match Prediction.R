#Author- Chandrasen Wadikar
#Date - 14/07/2018
#Prediction of team which would perform better in comming match 

#Required packages installed

install.packages('readxl')
library(readxl)

#Reading data set
matchdata<-read.csv('C:/Users/Chandrasen1/Desktop/matches.csv')
matchdata
#View dataset
View(matchdata)
#Replace NA,NAN with Mean
mean(matchdata)
matchdata$win_by_runs
vect<-c(1,2,3,NA,NaN)
is.na(vect)
is.nan(vect)

View(matchdata)
#Check dimension
dim(matchdata)

for (j in names(data)) setattr(data[[j]],"levels",{
  z <- levels(data[[j]])
  z[z=="NULL"] <- NA
  z
})

#Develop the model
matchlm<-lm(season ~ win_by_runs+win_by_wickets+city+team1+team2+toss_decision+toss_winner+winner+player_of_match, data = matchdata)
matchlm


#Check the result in summary, more focus on Adjusted R sequare and R sequare and analyzing the
# resulting with other parameters
summary(matchlm)
#Plotting the model
plot(matchlm)


fitmodel=data.frame(predict(matchlm,interval="prediction"))
fitmodel



