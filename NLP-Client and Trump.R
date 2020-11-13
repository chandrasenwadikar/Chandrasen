#############TEXT ANALYSIS on CLINTON SPEECH##########################

###Wordcloud analysis

install.packages("tm")#Text mining 
library(tm)
install.packages("SnowballC")#For stemming
library(SnowballC)
install.packages("stringr")#For split
library(stringr)
install.packages("wordcloud")#For wordcloud
library(wordcloud)

aa<-readLines("C:/Users/chandrasen.wadikar/Desktop/clinton_3.txt")
aa

#text<-paste(readLines("C:/Users/chandrasen.wadikar/Desktop/clinton_3.txt"),collapse =" ")

text1<-paste(readLines("C:/Users/chandrasen.wadikar/Desktop/clinton_3.txt"),collapse = " ")

text3<-paste(readLines("C:/Users/chandrasen.wadikar/Desktop/clinton_3.txt"),collapse = " ")

print(stopwords())#Find the stop words in document

text2<-removeWords(text,stopwords()) #removing stop words

bag_of_word1<-str_split(text2," ")#Split the words , if there is any space

str(bag_of_word1)

bag_of_word1<-unlist(bag_of_word1) #to unlist it

str(bag_of_word1) #to check if it is unlist

wordcloud(bag_of_word1,min.freq = 5,random.order = FALSE) # Wordcloud created

#Remove Words

text2<-removeWords(text2,c("this","poor","let","now"))

length(bag_of_word1)

#Sentiment Analysis

install.packages("syuzhet")
library(syuzhet)
install.packages("plyr")
library(plyr)
install.packages("sentimentr")
library(sentimentr)


mysent<-get_nrc_sentiment(text2)
mysent

ab=as.matrix(mysent)

barplot(ab,main = 'Clinton Speech Sentiment',xlab = 'Sentiment Breakup',ylab = 'Score',col = c('Orange'))
 

#############TEXT ANALYSIS on Trump SPEECH##########################

###Wordcloud analysis

install.packages("tm")#Text mining 
library(tm)
install.packages("SnowballC")#For stemming
library(SnowballC)
install.packages("stringr")#For split
library(stringr)
install.packages("wordcloud")#For wordcloud
library(wordcloud)

aa<-readLines("C:/Users/chandrasen.wadikar/Desktop/Trump_Austin_Aug-23-16.txt")
aa

#text<-paste(readLines("C:/Users/chandrasen.wadikar/Desktop/clinton_3.txt"),collapse =" ")

text1<-paste(readLines("C:/Users/chandrasen.wadikar/Desktop/clinton_3.txt"),collapse = " ")

text3<-paste(readLines("C:/Users/chandrasen.wadikar/Desktop/clinton_3.txt"),collapse = " ")

print(stopwords())#Find the stop words in document

text2<-removeWords(text,stopwords()) #removing stop words

bag_of_word1<-str_split(text2," ")#Split the words , if there is any space

str(bag_of_word1)

bag_of_word1<-unlist(bag_of_word1) #to unlist it

str(bag_of_word1) #to check if it is unlist

wordcloud(bag_of_word1,min.freq = 5,random.order = FALSE) # Wordcloud created

#Remove Words

text2<-removeWords(text2,c("this","poor","let","now"))

length(bag_of_word1)

#Sentiment Analysis

install.packages("syuzhet")
library(syuzhet)
install.packages("plyr")
library(plyr)
install.packages("sentimentr")
library(sentimentr)


mysent<-get_nrc_sentiment(text2)
mysent

ab=as.matrix(mysent)

barplot(ab,main = 'Trump Speech Sentiment',xlab = 'Sentiment Breakup',ylab = 'Score',col = c('Orange'))
