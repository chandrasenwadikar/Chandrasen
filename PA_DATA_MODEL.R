## PA Data Analysis ###
### Author : Chandrasen 
### Date : 22-March-2018

# Installing required libraries to read the data

install.packages('readxl')
library(readxl)

#Extracting and reading the data into vector

Pa<-read.csv("C:/Users/chandrasen.wadikar/Desktop/PA data.csv")

View(Pa) #View the data and validate whether it is now stored into Pa.

#Define the varibale names in machine learning 

colnames(Pa)[1]<-"Project_Name"
colnames(Pa)[2]<-"PM_Lead"
colnames(Pa)[3]<-"Defects_by_PA"
colnames(Pa)[4]<-"Defects_by_UAT"
colnames(Pa)[5]<-"Defect_lek"
colnames(Pa)[6]<-"functional_defects"
colnames(Pa)[7]<-"Percen_functional_def"
colnames(Pa)[8]<-"Total_open_defects"
colnames(Pa)[9]<-"Defect_re_open"
colnames(Pa)[10]<-"Total_tc"
colnames(Pa)[11]<-"Time_taken_tc"
colnames(Pa)[12]<-"Total_tc_exectuted"
colnames(Pa)[13]<-"Time_taken_exec"
colnames(Pa)[14]<-"Total_RA"
colnames(Pa)[15]<-"test_RA"
colnames(Pa)[16]<-"RA_Coverage"

head(Pa)
length(Pa)
summary(Pa)
#boxplot(Pa,horizontal = T)
#r1<-Pa[Pa<500]
#r1
#boxplot(r1,horizontal = T)
#hist(r1)
#test21 <-Pa[Pa<]

#bench<-58.0+1.5*IQR(Pa)
# Model-1 

pa_model1 <- lm(Defects_by_PA ~Defects_by_UAT+Defect_lek+Total_open_defects+Defect_re_open+Total_tc+Time_taken_exec+Total_RA+RA_Coverage,data = Pa)
summary(pa_model1)
plot(pa_model1)

predict_test1<-predict(pa_model1,Pa)
predict_actual_pa<-data.frame(predict_test1,Pa)
predict_actual_pa

??outlier

outlier(72,65)

#Output

#Coefficients:

#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept)        20.203477  31.749729   0.636 0.528095    
#Defects_by_UAT      9.023746   0.524740  17.197  < 2e-16 ***
 # Defect_lek         -1.253687   0.325144  -3.856 0.000399 ***
  #Total_open_defects  0.233136   0.544373   0.428 0.670699    
#Defect_re_open     -0.133097   0.121168  -1.098 0.278417    
#Total_tc            0.004997   0.031166   0.160 0.873404    
#Time_taken_exec     0.054533   0.076452   0.713 0.479700    
#Total_RA            0.503121   0.712820   0.706 0.484293    
#RA_Coverage        -0.090049   0.317830  -0.283 0.778352    
---
 # Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 19.72 on 41 degrees of freedom
#(27 observations deleted due to missingness)
#Multiple R-squared:  0.9981,	Adjusted R-squared:  0.9978 
#F-statistic:  2727 on 8 and 41 DF,  p-value: < 2.2e-16 
  
  
  
#Model-2 with different analysis
  
pa_model2 <- lm(Defects_by_PA ~Defects_by_UAT+Defect_lek+functional_defects+Percen_functional_def+Total_open_defects+Defect_re_open,data = Pa)
cooksd<-cooks.distance(pa_model2)
summary(pa_model2)
plot(pa_model2)


data(Pa)







predict_test2<-predict(pa_model2,Pa)
predict_actual_pa_model2<-data.frame(predict_test2,Pa)
predict_actual_pa_model2
  


