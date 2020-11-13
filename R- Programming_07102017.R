#PIE Chart

GDP <-c(19.4,9,23.8,33.9,3,5,6,7.9)
mean(GDP)
countries <-c("UK","India","Japan","Korea","USA")
GDP_PIE <-pie(GDP,labels = countries,main="gdp distribution")

sum_gdp <- sum(GDP)

perc_gdp <-round((GDP /sum_gdp)*100)

perc_count_label <-paste(countries,perc_gdp,'%')

GDP_PIE_PER<-pie(GDP,labels=perc_count_label,main="GDP distribution in % of countires")


empdetails <-c("Name","salary")
rating <-c(1,2,3,4,9)
result <-pie3D(empdetails,lables=rating,main="result123")

install.packages("plotrix")
library (plotrix)

GDP_PIE_PER<-pie3D(GDP,labels=perc_count_label,main="GDP distribution in % of countires")


#Box Plot  shows the minimum,maximum,median,1st quartile and 3rd quartile 

boxplot(mtcars $disp,main='Cylinder Summary')

boxplot(emp_sal $salary, main ='Details')

#HeatMap

matt_cars <-as.matrix(mtcars)
heatmap(matt_cars,col=heat.colors(256),Rowv=NA,Colv=NA,scale='column')

#User defined functions

add_num <- function  (a1,a2){
  a3 <-a1*a2  # a3 is local variable here , because it is defined inside the body
  return (a3)
}
add_num(12,10)


aa_num <- function (a1,a2){
  a3 <<- a1+a2 # to access the value outside the function use << -
  ab = a1-a2
  print(a3)
  return(a3)
}

aa_num(2,23)

bb_num <- function (a1,a2,a3,a4){
  a5 <<- a1+a2+a3+a4
  return(a5)
}
bb_num(10,20,13,29)

mode_test <- function(m1,m2,m3,m4){
  m2 <<- mode(c(m1,m2,m3,m4))
  return(m2)
}
mode_test(90,23,34,88)

mode_test <- function(m1,m2,m3,m4){
  m2 <<- mean(c(m1+m2+m3+m4)/4)
  return(m2)
}
mode_test(90,23,34,88)


bb_factorial <- function (a1,a2,a3){
  testf<<- factorial(c(a1,a2,a3))
  return(testf)
}
bb_factorial(1,5,3)  


#Train and Test for Build and test algorithm

irisdata <- iris #Sample remains the same when using the set.seed function
irissample1 <- sample(2,nrow(irisdata),replace=TRUE,prob = c(.7,.3)) #prob =probability
trainiris1 <- irisdata[irissample1==1,]# after 1 comma --indicates to select all data
testiris1 <- irisdata[irissample1==2,]# after 2 comma --indicates to select all data


#Write Function
aa <-c (1,2,3,4,5)
write.csv(aa,"C:/Users/Administrator/Desktop/Chandrasen/HospitalCoststest.csv")

xamp_text_file <-('this will be created as a text file for test')
#writeLines(xamp_text_file,"path/testing.txt")
writeLines(xamp_text_file,"C:/Users/Administrator/Desktop/Chandrasen/testing.txt")

#Cbind and rbind -- c bind used for add your columns and r bind is for add rows

aa <-c(9,4,5,6,10,11);mat_aa <-matrix(aa,2,3);mat_aa #create matrix
new_col <-c(77,88)
exm_bind <-cbind(mat_aa,(new_col));

aa <-c(9,4,5,6,10,11);mat_aa <-matrix(aa,2,3);mat_aa #create matrix
new_row <-c(77,88)
exm_bind <-rbind(mat_aa,(new_row));



