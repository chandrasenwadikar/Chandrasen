#Decision Tree Algorithm on AMES Housing Data Set
#Author : Chandrasen
#Date : 24-Apr-2018


################################ LINEAR REGRESSION MODEL #####################################
#Install required packages to read the CSV 

install.packages('readxl')
library(readxl)

#Assign Vector to CSV 

housinglinear<-read.csv("C:/Users/chandrasen.wadikar/Desktop/Housing.csv")
housinglinear

#Viewing the Data Set

View(housinglinear)

#Converting NA/NULL into Mean for identified columns

mean(housinglinear)
housinglinear$LotFrontage
vect<-c(1,2,3,NA,NaN)
is.na(vect)
is.nan(vect)


LotFrontage.mean <- mean(housinglinear$LotFrontage, na.rm=TRUE)
housinglinear$LotFrontage[is.na(housinglinear$LotFrontage)] = LotFrontage.mean
LotFrontage.mean

View(housinglinear)

#Converting NA/NULL into Mean for identified columns

mean(housinglinear)
housinglinear$MasVnrArea
vect<-c(1,2,3,NA,NaN)
is.na(vect)
is.nan(vect)


MasVnrArea.mean <- mean(housinglinear$MasVnrArea, na.rm=TRUE)
housinglinear$MasVnrArea[is.na(housinglinear$MasVnrArea)] = MasVnrArea.mean
MasVnrArea.mean

View(housinglinear$MasVnrArea)

#Rename the column name and make it logical

names(housinglinear)[43]<-"SecondndFlrSF"

View(housinglinear)

#Check the dimesion

dim(housinglinear)


#Convert the NULL into NA

#f=function(x){
 # x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  #x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  #x #display the column
#}
#ss=data.frame(apply(df,2,f))

#Replacing Null with NA

for (j in names(data)) setattr(data[[j]],"levels",{
  z <- levels(data[[j]])
  z[z=="NULL"] <- NA
  z
})


#Developing the model



#housinglin<-lm(SalePrice~MSSubClass+LotArea+LandContour+Utilities+LotConfig+Neighborhood+BldgType+HouseStyle+OverallQual+OverallCond
  #             +YearBuilt+MasVnrType+Foundation+BsmtCond+BsmtFinType1+BsmtFinSF1+TotalBsmtSF+Heating+FullBath+HalfBath+KitchenQual
   #            +GarageCars+GarageArea+GarageCond+SaleType+SaleCondition,data = housinglinear)


housinglin<-lm(SalePrice~ MSSubClass+LotArea+LotShape+LandContour+LotConfig+Neighborhood+Condition2+BldgType
               +OverallQual+OverallCond+YearBuilt+Exterior1st+MasVnrType+MasVnrArea+Foundation+BsmtFinSF1+TotalBsmtSF+Heating++Electrical
               +GrLivArea+FullBath+HalfBath+BedroomAbvGr+KitchenQual+Fireplaces+Functional
               +GarageCars+GarageArea+SaleType+SaleCondition, data = housingdata)


housinglm<-lm(SalePrice~ MSSubClass+LotFrontage+LotArea+OverallQual+OverallCond+YearBuilt+MasVnrArea+BsmtFinSF1+BsmtFinSF2+BsmtUnfSF
              +TotalBsmtSF+GrLivArea+FullBath+HalfBath+BedroomAbvGr+TotRmsAbvGrd+Fireplaces+GarageCars+GarageArea,data = housinglinear)


housinglm<-lm(SalePrice ~ MSSubClass + LotArea + LandContour + 
                Utilities + LotConfig + Neighborhood + BldgType + 
                HouseStyle + OverallQual + OverallCond + YearBuilt +  
                MasVnrType + Foundation + BsmtCond + 
                BsmtFinType1 + BsmtFinSF1 + TotalBsmtSF + Heating + 
                X1stFlrSF + X2ndFlrSF + FullBath + HalfBath + KitchenQual + 
                GarageCars + GarageArea + GarageCond + SaleType + 
                SaleCondition, data = housinglinear)


summary(housinglm)

plot(housinglm)


fitmodel=data.frame(predict(housinglm,interval="prediction"))



head(housinglinear)
length(housinglinear)

hist(housinglinear)

boxplot(housinglinear,horizontal = T)



##Exporting Data set ####

write.csv(housinglinear, "C:/Users/chandrasen.wadikar/Desktop/housinglinear.csv")

#############Principle Component Analysis ###############################

mydata<-read.csv("C:/Users/chandrasen.wadikar/Desktop/housinglinear.csv")
attach(mydata)

x<-cbind(MSSubClass,LotArea,OverallQual,OverallCond,YearBuilt,MasVnrArea,BsmtFinSF1,BsmtUnfSF,TotalBsmtSF,X1stFlrSF,X2ndFlrSF,GrLivArea,FullBath,HalfBath,TotRmsAbvGrd,GarageCars,GarageArea,WoodDeckSF,data = housinglinear)
summary(x)
cor(x)

pca1<-princomp(x,scores = TRUE,cor = TRUE)
summary(pca1)

loadings(pca1)

plot(pca1)
screeplot(pca1,type = "line",main = "Scree Plot")
biplot(pca1)

pca1$scores[1:10,]



#####################################CHI- sequare #####################################

install.packages('readxl')
library(readxl)

housing <- read.csv("C:/Users/chandrasen.wadikar/Desktop/housinglinear.csv")
housing

View(housing)

install.packages('MASS')
library(MASS)

housing_table <- table(housing$SalePrice,housing$MSSubClass,housing$LotArea,housing$LandContour,housing$Utilities,housing$LotConfig,housing$Neighborhood,housing$BldgType, 
                       housing$HouseStyle,housing$OverallQual,housing$OverallCond,housing$YearBuilt,housing$MasVnrType,housing$Foundation,housing$BsmtCond, 
                       housing$BsmtFinType1,housing$BsmtFinSF1,housing$TotalBsmtSF,housing$Heating,housing$X1stFlrSF,housing$X2ndFlrSF,housing$FullBath,housing$HalfBath,housing$KitchenQual,
                       housing$GarageCars,housing$GarageArea,housing$GarageCond,housing$SaleType,housing$SaleCondition)



############################################K-MEANS################################################


install.packages("dplyr")
library(dplyr)

install.packages('readxl')
library(readxl)

kmeansalgo <- read.csv("C:/Users/chandrasen.wadikar/Desktop/housinglinear.csv")
kmeansalgo

kmeansalgo<-na.omit(kmeansalgo)

View(kmeansalgo)

kmeansalgo<-select(housinglinear - SalePrice.mean)

model_housing_kmeans<-kmeans(kmeansalgo,3,nstart = 1,iter.max = 50)
model_housing_kmeans

View(kmeansalgo)


############################ NAVIE BAYES ##############################

install.packages('readxl')
library(readxl)

housing_nb <-read.csv("C:/Users/chandrasen.wadikar/Desktop/housinglinear.csv")
housing_nb

View(housing_nb)

install.packages("e1071")
library(e1071)
install.packages("caret")
library(caret)


navie_bayes_model=navieBayes(SalePrice ~., data = housinglinear)


##################Random Forest ##########################

install.packages('randomForest')
library(randomForest)

install.packages('readxl')
library(readxl)

hl <- read.csv("C:/Users/chandrasen.wadikar/Desktop/housinglinear.csv")
hl

install.packages('rpart')
library(rpart)
install.packages('party')
library(party)

hl$SalePrice<-factor(hl$SalePrice)
is.factor(hl$SalePrice)

model_housing_rf<-randomForest(SalePrice ~ MSSubClass+LotFrontage+LotArea+OverallQual+OverallCond+YearBuilt+MasVnrArea+BsmtUnfSF+TotalBsmtSF+
                                 X1stFlrSF+X2ndFlrSF+GrLivArea+BsmtFullBath+FullBath+HalfBath+BedroomAbvGr+KitchenAbvGr+GarageCars+GarageArea+
                                 WoodDeckSF+OpenPorchSF+MiscVal+YrSold, data = housinglinear, controls= ctree_control(mincriterion = .90,minsplit = 500))
model_housing_rf

plot(model_housing_rf)

hl_sample_rf <- sample(2,nrow(hl),replace = TRUE,prob = c(.8,.2))
hl_train_rf<-hl[hl_sample_rf==1,]
hl_test_rf<-hl[hl_sample_rf==2,]



hl_predict_rf<-predict(model_housing_rf,hl_test_rf,type="prob")
hl_predict_rf1<-predict(model_housing_rf,hl_test_rf,type="response")








