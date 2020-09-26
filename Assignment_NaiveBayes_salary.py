################# ASSIGNMENT NAIVE BAYES salary-dataset

# Problem statement: To understand the salary dataset and build  NAIVE BAYES 
# CLASSIFIER to predict if the individual belongs to low income or high income
# group (<=50K  or >50K)

## importing required libraries
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from mixed_naive_bayes import MixedNB # first do "pip install mixed_naive_bayes"
from sklearn.preprocessing import LabelEncoder

## importing dataset for training the model
salary_train_raw = pd.read_csv("E:/NaiveBayes/SalaryData_Train.csv")

salary_train_raw.shape # 30161 records, 14 variables
salary_train_raw.head()
salary_train_raw.columns

##### missing values details
salary_train_raw.isnull().sum() # no missing values

# summary statistics of continuous and categorical variables
salary_train_raw.describe() # stats of continuous vars
# salary_train_raw.describe(include=["number"])
salary_train_raw.describe(include=["object"]) # stats of categorical variables

# finding the data type of variables
salary_train_raw.dtypes
# continuous variables : 
'age',  'educationno','capitalgain','capitalloss', 'hoursperweek'
       
# Categorical variables : 
'workclass', 'education','maritalstatus','occupation', 
'relationship', 'race', 'sex', 'native'

# outcome variable : Salary with 2 levels
# from collections import Counter
Counter(salary_train_raw.Salary)
# 2 levels: <=50K  and >50K
# {' <=50K': 22653,      ' >50K': 7508})

##### using validation data set
# already data is divided into train data for training the model
# and test data for testing the model
salary_test_raw = pd.read_csv("E:/NaiveBayes/SalaryData_Test.csv")

salary_test_raw.shape # 15060 records, 14 vars
salary_test_raw.head()
salary_test_raw.dtypes


################# encoding the predictor text classes to numbered-values
# from sklearn import preprocessing
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native","Salary"]
salary_train=salary_train_raw.copy()
salary_test = salary_test_raw.copy()

le = LabelEncoder()
for i in string_columns:
    salary_train[i] = le.fit_transform(salary_train[i])
    salary_test[i] = le.fit_transform(salary_test[i])

salary_train.dtypes
salary_test.dtypes
# now all columns have been converted to numeric type

### separating the dataset into predictors and outcome variables
colnames = salary_train.columns
Xtrain = salary_train[colnames[0:13]]
ytrain = salary_train[colnames[13]]
Xtest = salary_test[colnames[0:13]]
ytest = salary_test[colnames[13]]

################## Model selection and building 
# from sklearn.naive_bayes import GaussianNB, MultinomialNB

############# Model 1 using Gaussian NB
# we use Gausssian NB for continuous predictors
gauss_nb = GaussianNB()
gauss_pred = gauss_nb.fit(Xtrain,ytrain).predict(Xtest)

## Model evaluation
from sklearn import  metrics

metrics.accuracy_score(ytest, gauss_pred)
# Gaussian NB model accuracy is 0.7946879

metrics.confusion_matrix(ytest, gauss_pred)
#     [[10759,   601],
#     [ 2491,  1209]]

############# Model 2 using Multinomial NB
# we use MUltinomial NB for categorical predictors
multinomial_nb  = MultinomialNB()
multinomial_pred = multinomial_nb.fit(Xtrain,ytrain).predict(Xtest)

metrics.accuracy_score(ytest, multinomial_pred)
# Multinomial NB model accuracy is 0.774966

metrics.confusion_matrix(ytest, multinomial_pred)
#    [[10891,   469],
#    [ 2920,   780]]

############# Model 3 using Mixed NB
# here we will use Gaussian NB for continuous predictors and MUltinomial NB
# for categorical predictors
# first install it using - pip install MixedNB
from mixed_naive_bayes import MixedNB
salary_train_raw.dtypes
mixed_model = MixedNB(categorical_features=[1,2,4,5,6,7,8,12])
mixed_model_pred = mixed_model.fit(Xtrain,ytrain).predict(Xtest)

metrics.accuracy_score(ytest,mixed_model_pred)
# 0.8242

# we can see that the mixed model has highest accuracy score

#################### Model 4 using Logistic regression

from sklearn.linear_model import LogisticRegression
sal_logreg = LogisticRegression()
sal_logreg.fit(Xtrain,ytrain)

logreg_pred = sal_logreg.predict(Xtest)

Counter(ytest) # {0: 11360, 1: 3700}
pd.crosstab(ytest,logreg_pred)

metrics.accuracy_score(ytest,logreg_pred)  # 0.8038
metrics.precision_recall_fscore_support(ytest,logreg_pred)
'''
(array([0.83246322, 0.65425972]), precision is 0.832
 array([0.92640845, 0.42756757]), sensitivity is 0.926 and specificity is 0.428
 array([0.87692692, 0.51716247]), '''

metrics.accuracy_score(ytest,gauss_pred)  # 0.7946
metrics.precision_recall_fscore_support(ytest,gauss_pred) 
'''
(array([0.812    , 0.6679558]), precision is 0.812 
 array([0.94709507, 0.32675676]) sensitivity is 0.9470 and specificity is 0.3267 '''

metrics.accuracy_score(ytest,multinomial_pred)  # 0.774966
metrics.precision_recall_fscore_support(ytest,multinomial_pred) 
'''
(array([0.78857432, 0.6244996 ]), precision is 0.788
 array([0.95871479, 0.21081081]), sensitivity is 0.9587 and specificity is 0.2108
'''
metrics.accuracy_score(ytest,mixed_model_pred) #  0.82423
metrics.precision_recall_fscore_support(ytest,mixed_model_pred)
'''
array([0.85175616, 0.69682243]), precision is 0.85175
 array([0.92860915, 0.50378378]), sensitivity is 0.9286 and specificity is 0.50378
'''

''' The mixed naive-bayes model shows the highest accuracy, precision and specificity.
Highest sensitivity is given by multinomial naive-bayes model. 
Now let us look into the Mixed naive bayes model metrics:
    This model has correctly predicted about 82% of the cases (accuracy).
    Of those that have been predicted as <=50K by the model, 85% are actually <=50K . 
    when salary is actually <=50K, it has correctly predicted 92.9% (sensitivity).
    when salary is actually >50K, it has correctly predicted only 50.4% (specificity).

'''
'''
CONCLUSIONS

We have to understand the salary dataset and predict to which class an employee
belongs: whether his salary is less than or equal to 50K or more than 50K. The
data has both continuous and categorical predictors. Since the categorical data 
is in text form, we have first encoded them using labelencoder. 

We have used different Naive Bayes methods to classify the data:
    First we have used Gaussian NB, which assumes preditors to be normally 
    distributed. we have got accuracy of about 79%. 
    Next, we used Multinomial NB, which assumes predictors to have multinomial 
    distribution, rather than some other distribution. Here we have obtained 
    around 77% accuracy.
    Since our dataset has both continuous and categorical predictors, we cannot
    assume one type of distribution for all data types. So next we have used
    Mixed NB, which assumes normal distribution for continuous variables and
    multinomial distribution for categorical variables. Here we obtained
    highest accuracy of 82%, highest precision (85%) and highest specificity (50%).

We have also used Logistic regression to see if it gives better results than
the Naive Bayes' models. 

For the salary dataset, we have found that mixed Naive Bayes gives better results
compared to other models.

'''