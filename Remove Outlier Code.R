data("rivers")
data
head(rivers)
length(rivers)

hist(rivers)
boxplot(rivers,horizontal = T)

r1<-rivers[rivers<500]
View(r1)
r1<1500

boxplot(r1,horizontal = T)

