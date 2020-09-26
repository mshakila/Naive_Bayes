######## ASSIGNMENT NAIVE BAYES using salary data


# PROBLEM STATEMENT: To study salary dataset and build a Machine learning model
# that predicts whether an person's salary is <=50k or it is >50K


### Data set Description:
'''
The given dataset contains about 30,000 observations of individuals along
with their demographic and employment details. 

Below is a list of the predictor variables which will help us to classify
whether a person"s salary is less or more than 50,000

age -- age of a person
workclass	-- A work class is a grouping of work 
education	-- Education of an individuals	
maritalstatus -- Marital status of an individulas	
occupation	 -- occupation of an individuals
relationship -- if married or unmarried. If married whether person is 
  husband or wife and similar details
race --  Race of an Individual
sex --  Gender of an Individual
capitalgain --  profit received from the sale of an investment	
capitalloss	-- A decrease in the value of a capital asset
hoursperweek -- number of hours work per week	
native -- Native of an individual
Salary -- salary of an individual
'''

#### importing required libraries
library(psych) # many functions including describe()
library(Amelia) # to plot missing values in a dataset
library(ggplot2) # visualizations, ggplot
library(GGally) # to get plot of correlation
library(e1071) # ro run naive bayes model
library(caret) # for confusionmatrix function
library(gmodels) # for crosstable function

# importing salary dataset that will be used to train the model

salary_train_raw <- read.csv("E:\\NaiveBayes\\SalaryData_Train.csv")
# abt 30,000 records and 14 variables
head(salary_train_raw)
names(salary_train_raw)
# [1] "age"           "workclass"     "education"     "educationno"   "maritalstatus"
# [6] "occupation"    "relationship"  "race"          "sex"           "capitalgain"  
# [11] "capitalloss"   "hoursperweek"  "native"        "Salary" 


# studying structure of dataset
str(salary_train_raw)
'''
there are 5 variables of continuous data type viz, age,educationno, capitalgain,
capitalloss, hoursperweek. Remaining 9 are categorical variables. 
The outcome variable will be Salary with 2 levels :
1 - indicates salary is <= 50K
2 - indicates salary is >50K '''

# salary$Salary <- factor(salary$Salary, levels=c(0,1), labels=c('<=50K','>50K'))

#### Understanding the dataset
head(salary_train_raw)

# finding proportion of Salary class
table(salary_train_raw$Salary)
prop.table(table(salary_train_raw$Salary))
#     <=50K   >50K 
#     22653   7508
# 0.7510693 0.2489307
# about 75% have salary <=50k and 25% have salary >50k

# finding missing values
sum(is.na(salary_train_raw)) # no missing values
# Naive Bayes itself treats missing values 


###### Do we need to standardize the data ?
# naive bayes internally does scaling of data. Hence no need to standardize

# library(psych)
describe(salary_train_raw[,c(1,4,10,11,12)])

describe(salary_train_raw$capitalgain)
'''
capitalgain: we see that penultimate value is 41,310 and max value is 99,999
with 148 records. The wide gap only indicates that the values 99,999 are missing 
values. Converting these to NA '''
salary_train <- salary_train_raw
salary_train[,10][salary_train[,10] == 99999] <- NA
describe(salary_train[,c(1,4,10,11,12)])
sum(is.na(salary_train))
# now there are 30013 valid entries in capitalgain and 148 missing values

# visualize the missing data
# library(Amelia)
missmap(salary_train)
# white lines in capital gain show there are some missing entries in that variable
# this (149 out of 30161 entries) accounts for hardly 0.005% , 
# hence missing data given as 0%
# removing the missing data entries
salary_train_new <- na.omit(salary_train) # now there are 30013 entries
missmap(salary_train_new) # fully blue (and no white lines), indicating no missing values
sum(is.na(salary_train_new$capitalgain)) # now no missing values
names(salary_train_new)
# attach(salary_train_new)

###### Data visualization

# visual 1: Distribution of Salary class vs Age of employees
# library(ggplot2)
A <- ggplot(salary_train, aes(age, fill=Salary, colour = Salary)) +
  geom_histogram(binwidth = 1) + 
  labs(title="Age Distribution by Salary variable")
A + theme_bw()
'''
age ranges from 17 years to 90 years
for salary of <=50k, there are more persons btw 19 to 50 years
for salary of >50k, there are more persons btw 34 to 52 years
'''
# visual 2: Distribution of Salary class vs Number-of-education-years
ggplot(salary_train, aes(educationno, colour = Salary)) +
  geom_freqpoly(binwidth = 1) + 
  labs(title="Number of years of Education by Salary variable")
''' 
Number of years of education ranges btw 1 to 16 years
When salary is <=50K, there are many persons between  8 to 11 years of education.
In >50K level till 8years of eduction there are no persons. There are
more persons btw 8 to 10 years and between 12 to 13 years of education '''

# visual 3: Distribution of Salary class vs Hours-worked-per-week
ggplot(salary_train, aes(hoursperweek, colour = Salary)) +
  geom_freqpoly(binwidth = 1) + labs(title="Hours-per-week Distribution by Salary variable")
''' in both categories, there are more persons who work for about 37 to 42
hours-per-week, the count of <=50k is much more than >50k . 
There are persons in both categories who work even upto 99 hours/wk'''

# library(GGally)
ggpairs(salary_train[,c(1,4,10,11,12)])
'''
the plot shows that there is almost no linear relationship between
continuous variables. Naive bayes considers that the predictors are independent
( hence the name naive) '''


######## DATA MODELLING
### data splitting:
# the data is already divided into training and testing set
## Training set: we will use salary_train (pre-processed data) as training dataset

## Testing set: importing test dataset
salary_test_raw <- read.csv("E:\\NaiveBayes\\SalaryData_Test.csv")
describe(salary_test_raw) # there are 15060 records


###### Modelling the data using the raw (unchanged) train and test data
model_sal_raw <- naiveBayes(salary_train_raw$Salary~., data=salary_train_raw[,-14])
pred_sal_raw <- predict(model_sal_raw, salary_test_raw[,-14])

# pred_train_raw <- predict(model_sal_raw,salary_train_raw[,-14])
# confusionMatrix(pred_train_raw,salary_train_raw[,14])

mean(pred_sal_raw == salary_test_raw[,14])
# overall accuracy of 81.93%

# library(caret)
confusionMatrix(pred_sal_raw,salary_test_raw[,14], dnn = c("predicted","actual"))
#             actual
# predicted  <=50K  >50K
# <=50K     10550  1911
# >50K       810  1789

CrossTable(pred_sal_raw,salary_test_raw$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted','actual'))
# we can see that in <=50K class, about 92.9% are correctly predicted
# but in >50K class only about 48.4% are correctly predicted
# the overall accuracy of 81.93% is thus misleading.


########### MODEL BY making changes to "capitalgain" predictor (na.action=na.pass)
#  (changing 99999 to NA)

# capitalgain variable: converting 99999 into NA, there are 81 such entries
salary_test <- salary_test_raw
salary_test[,10][salary_test[,10] == 99999] <- NA
missmap(salary_test)
sum(is.na(salary_test)) # 81 NAs

# salary_test <- na.omit(salary_test) # removing missing value from capitalgain variable
# describe(salary_test) # removing 81 NA, get 14979 records

# train dataset: salary_train
# test dataset: salary_test

prop.table(table(salary_train$Salary))
prop.table(table(salary_test$Salary))
# the proportion of both levels is same in both train and test data.
# There are 75% records with <=50K salary and 25% records with salary >50K

model_sal1 <- naiveBayes(salary_train$Salary ~., data=salary_train[,-14],na.action = na.pass)
pred_sal1 <- predict(model_sal1, salary_test[,-14])
confusionMatrix(pred_sal1,salary_test[,14], dnn = c("predicted","actual"))
# na.pass, this will ignore the NA value
CrossTable(pred_sal1,salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted','actual'))
# overall accuracy is 82%, 
# 90% in <=50K class and 58% in >=50K class are correctly predicted
# though specificity of this model is far better than previous model, overall
# accuracy is very low compared to previous model.

################### Model by making changes to "capitalgain" predictor (using na.action = na.omit)

model_sal2 <- naiveBayes(salary_train$Salary ~., data=salary_train[,-14],na.action = na.omit)
# na.omit, this will ignore the complete record where NA is found

pred_sal2 <- predict(model_sal2, salary_test[,-14])
confusionMatrix(pred_sal2,salary_test[,14], dnn = c("predicted","actual"))

# here accuracy, sensitivity and specificity are same as model_sal1

############### Model (using na.action = na.omit and laplace=1)

model_sal3 <- naiveBayes(salary_train$Salary ~., data=salary_train[,-14],
                         na.action = na.omit,laplace = 1)
pred_sal3 <- predict(model_sal3, salary_test[,-14])
confusionMatrix(pred_sal3,salary_test[,14], dnn = c("predicted","actual"))
# here accuracy, sensitivity and specificity are same as model_sal1 and model_sal2


######## Model by removing missing values
salary_train_new <- na.omit(salary_train) 
# removing missing value from capitalgain variable, 30161-148=30013 
describe(salary_train_new) # removing 148 NA, get 30013 records

salary_test_new <- na.omit(salary_test) 
# removing missing value from capitalgain variable. 15060-81=14979
describe(salary_test_new) # removing 81 NA, get 14979 records

model_sal4 <- naiveBayes(salary_train_new$Salary ~ ., data=salary_train_new[,-14])
pred_sal4 <- predict(model_sal4, salary_test_new[,-14])
confusionMatrix(pred_sal4, salary_test_new[,14], dnn=c("predicted","actual"))
# here accuracy, sensitivity and specificity are very similar to model_sal1 ,
# model_sal2 and model_sal3


######## Model by removing outliers
########## outliers detection and removing them
# NB uses normal distribution for continuous predictors. outliers affect the
# shape of normal distribution (for example, mean gets affected). Hence,
# NB is sensitive to outliers. So we need to detect and remove them. we will 
# be using the salary_train data where we have made changes to "capitalgain" variable

hist(salary_train$educationno)
boxplot(salary_train$educationno, horizontal = T)
boxplot.stats(salary_train$educationno)$out # 196 obs

a <- boxplot(salary_train$age,horizontal = T)
boxplot.stats(salary_train$age)$out # 169 obs
min(a$out) # 76
max(a$out) # 90
range(salary_train$age) # 17 to 90


# removing  outliers of "age" in salary_train data
salary_train_ageNoOutlier <- salary_train[!rowSums(salary_train[1] >= 76),]
dim(salary_train_ageNoOutlier) # 29992 records (removed 169 records)
30161-29992 # 169
boxplot.stats(salary_test$age) # test data has 57 outliers


boxplot(salary_train$capitalgain,horizontal = T)
boxplot.stats(salary_train$capitalgain)$out # 

boxplot(salary_train$capitalloss,horizontal = T)
boxplot.stats(salary_train$capitalloss)$out # 

boxplot(salary_train$hoursperweek,horizontal = T)
boxplot.stats(salary_train$hoursperweek)$out # 
# there are many outliers in all 3 capgain, caploss, hours/wk. 
# so cannot remove these outliers

## model after removing outliers from "Age" predictor
model_sal_ageNoOutlier <- naiveBayes(salary_train_ageNoOutlier$Salary~., 
                                     data=salary_train_ageNoOutlier[,-14],laplace=1)
pred_sal_ageNoOutlier <- predict(model_sal_ageNoOutlier, salary_test[,-14])

confusionMatrix(pred_sal_ageNoOutlier,salary_test[,14],dnn = c('predicted','actual'))

# The model better than all model built is this model viz, model_sal_ageNoOutlier.
# Here, we have changed
# the capitalgain values (99999 to NA), used na.omit to ignore the records having
# missing values. Also, we have used laplace smoother as 1. 
# For this model :
# overall accuracy is 82.67%
# for salary <=50k, 90.69% have been correctly predicted (true positive rate or sensitivity)
# for salary >50k, 58.05% have been correctly predicted (true negative rate or specificity)


############### Logistic Regression
# Let us see if NB models obtained are better than logistic regression model
# we will use salary_train_new and salary_test_new data to build the model

logreg_sal <- glm(salary_train_new$Salary ~ ., data=salary_train_new, family="binomial")
summary(logreg_sal)

logreg_prob <- predict(logreg_sal, salary_test_new[,-14], type = "response")

confusion <- table(logreg_prob>0.5, salary_test_new$Salary,dnn = c('predicted','actual'))
confusion

Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy # 84.59
table(salary_test_new$Salary)
# <=50K   >50K 
# 11360   3619 
# 
10533/11360 # 0.9272007 is sensitivity
2138/3619 # 0.5907709 is specificity

# ROC Curve 
library(ROCR)
rocrpred<-prediction(logreg_prob, salary_test_new$Salary)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T)

# auc
auc <- performance(rocrpred,measure='auc')
auc <- auc@y.values[[1]]
auc # 0.8996525, here AUC is closer to 1, hence this is a good model

## when we use logistic regression model, for salary dataset, the accuracy, 
# sensitivity and specificity are slightly greater than our Naive bayes models. 
# Also the area-under-curve is around 90% which is very good.

# Logistic regression is a discriminative model. It learns from inputs and
# directly calculates the posterior probability P(Y/X1,X2,,Xn)

# Naive Bayes is a generative model, first it models the joint distribution of
# Xs and Y, it then calculates the posterior probability P(Y/X1,X2,,Xn)


'''
CONCLUSIONS:

We have used the salary dataset and built Naive Bayes model. Here we are
predicting the respondents into the 2 main salary classes (<=50k salary 
and >50k salary). The outcome variable is binary and hence we have used NB algorithm
to make predictions. We have built various models

We have used laplace smoothing technique to improve the performance.

We have also built model using another classification technique: Logistic regression.

This dataset has many outliers in the continuous variables which affect NB 
performance. 

'''


