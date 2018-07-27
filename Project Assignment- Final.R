
################################ LINEAR REGRESSION MODEL #####################################
#Install required packages to read the CSV 

install.packages('readxl')
library(readxl)

#Assign Vector to CSV 

housinglinear<-read.csv("C:/Users/Chandrasen1/Desktop/Housing.csv")
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

#housinglin<-lm(SalePrice~ MSSubClass+LotArea+LotShape+LandContour+LotConfig+Neighborhood+Condition2+BldgType
 #              +OverallQual+OverallCond+YearBuilt+Exterior1st+MasVnrType+MasVnrArea+Foundation+BsmtFinSF1+TotalBsmtSF+Heating++Electrical
  #             +GrLivArea+FullBath+HalfBath+BedroomAbvGr+KitchenQual+Fireplaces+Functional
   #            +GarageCars+GarageArea+SaleType+SaleCondition, data = housinglinear)


#housinglm<-lm(SalePrice~ MSSubClass+LotFrontage+LotArea+OverallQual+OverallCond+YearBuilt+MasVnrArea+BsmtFinSF1+BsmtFinSF2+BsmtUnfSF
#             +TotalBsmtSF+GrLivArea+FullBath+HalfBath+BedroomAbvGr+TotRmsAbvGrd+Fireplaces+GarageCars+GarageArea,data = housinglinear)


housinglm<-lm(SalePrice ~ MSSubClass + LotArea + LandContour + 
              Utilities + LotConfig + Neighborhood + BldgType + 
             HouseStyle + OverallQual + OverallCond + YearBuilt +  
            MasVnrType + Foundation + BsmtCond + 
            BsmtFinType1 + BsmtFinSF1 + TotalBsmtSF + Heating + 
          X1stFlrSF + X2ndFlrSF + FullBath + HalfBath + KitchenQual + 
         GarageCars + GarageArea + GarageCond + SaleType + 
         SaleCondition, data = housinglinear)


summary(housinglin)

plot(housinglin)


fitmodel=data.frame(predict(housinglin,interval="prediction"))



head(housinglinear)
length(housinglinear)

hist(housinglinear)

boxplot(housinglinear,horizontal = T)