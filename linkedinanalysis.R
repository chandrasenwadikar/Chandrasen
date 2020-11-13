install.packages("tm")
library(tm)
install.packages("SnowballC")#For stemming
library(SnowballC)
install.packages("stringr")#For split
library(stringr)
install.packages("wordcloud")#For wordcloud
library(wordcloud)

test<-readLines("C:/Users/chandrasen.wadikar/Desktop/linkedin.txt")
test

test12<-paste(readLines("C:/Users/chandrasen.wadikar/Desktop/linkedin.txt"),collapse = ' ')
print(stopwords())

aa<-removeWords(test12,stopwords())
aa

bag_of_word12<-str_split(aa," ")
str(bag_of_word12)

bag_of_word12<-unlist(bag_of_word12)
str(bag_of_word12)

wordcloud(bag_of_word12,min.freq = 5,random.order = FALSE)

#test12<-removeWords(test12,c("test, "experience"))

length(bag_of_word12)

install.packages("syuzhet")
library(syuzhet)
install.packages("plyr")
library(plyr)
install.packages("sentimentr")
library(sentimentr)

my<-get_nrc_sentiment(test12)
my

cc<-as.matrix(my)

barplot(cc,main = 'Summary of Profile',xlab = 'Different areas',ylab ='Score',col = c('Grey'))





######################################################################################33


install.packages("tm")
library(tm)
install.packages("SnowballC")#For stemming
library(SnowballC)
install.packages("stringr")#For split
library(stringr)
install.packages("wordcloud")#For wordcloud
library(wordcloud)

test<-readLines("C:/Users/chandrasen.wadikar/Desktop/linkedin.txt")
test

test12<-paste(readLines("C:/Users/chandrasen.wadikar/Desktop/manish.txt"),collapse = ' ')
print(stopwords())

aa<-removeWords(test12,stopwords())
aa

bag_of_word12<-str_split(aa," ")
str(bag_of_word12)

bag_of_word12<-unlist(bag_of_word12)
str(bag_of_word12)

wordcloud(bag_of_word12,min.freq = 5,random.order = FALSE)

#test12<-removeWords(test12,c("test, "experience"))

length(bag_of_word12)

install.packages("syuzhet")
library(syuzhet)
install.packages("plyr")
library(plyr)
install.packages("sentimentr")
library(sentimentr)

my<-get_nrc_sentiment(test12)
my

cc<-as.matrix(my)

barplot(cc,main = 'Summary of Profile',xlab = 'Different areas',ylab ='Score',col = c('Orange'))

 

