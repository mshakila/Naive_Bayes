######## ASSIGNMENT NAIVE BAYES using sms_data


# PROBLEM STATEMENT: To study sms_data and build a Machine learning model
# that classifies whether the given sms is ham or spam

# import required libraries
library(tm)
library(wordcloud)
library(e1071)
library(gmodels)
library(caret)

######## 1. read raw material into sms_raw
sms_data <- read.csv('E:\\NaiveBayes\\sms_raw_NB.csv',stringsAsFactors = FALSE)
head(sms_data,3)
class(sms_data)
str(sms_data) # text and type variable are of character type

######### 2. convert "type" variable from character to factor
sms_data$type <- factor(sms_data$type)
str(sms_data)

table(sms_data$type)
prop.table(table(sms_data$type))
# ham 86% and spam 13%

########### 3. create a corpus (sms_corpus) using text data (sms_raw$text)
# first we need to create a corpus which is collection of text documents
# library(tm)
sms_corpous <- Corpus(VectorSource(sms_data$text))
sms_corpous$content[1] # to view first document
sms_corpous$content[1:5] # to view multiple documents, first 5

# print(vignette("tm")) # tm package vignette gives details of corpus 
print(sms_corpous) # this corpus has 5559 documents
inspect(sms_corpous[1:5])

######### 4. cleaning data to remove punctuation, numbers, etc 
# clean the corpus to sms_corpus_clean by convert to lowercase, remove punctuation,
# remove number, remove filler word, stemming the word
corpus_clean <- tm_map(sms_corpous, tolower) # converting to lowercase
corpus_clean <- tm_map(corpus_clean,removeNumbers) # removing numbers
corpus_clean <- tm_map(corpus_clean,removeWords, stopwords()) # removing filler words (stopwords)
corpus_clean <- tm_map(corpus_clean, removePunctuation) # removing punctuations
corpus_clean <- tm_map(corpus_clean, stripWhitespace) 
# Removing additional whitespace, leaving only a single space between words.
# library(wordnet)
# corpus_clean <- lemmatize_words(corpus_clean)
# corpus_clean <- tm_map(corpus_clean, stripWhitespace) 

inspect(corpus_clean[1:6])

### visualizing cleaned corpus using word cloud
# library(wordcloud)
wordcloud(corpus_clean, min.freq = 60, random.order = FALSE, scale = c(3, 0.5))
# wordcloud(corpus_clean,min.freq = 60, random.order = F)
60/5559 # 1%
#  warnings()
# A frequency of 60 is about 1 percent of the corpus, this means that 
# a word must be found in at least 1 percent of the SMS messages to be included 
# in the cloud 

corpus_clean <- tm_map(corpus_clean, removeWords, c("wat",'cos','ish','also'))
# removing other words (as seen in wordcloud) that do not impart much meaning
corpus_clean <- tm_map(corpus_clean, stripWhitespace) 


########### 5. Splitting text documents into words (tokenizing) and creating matrix 
# creating a matrix sms_dtm by tokenizing the cleaned corpus. documents on rows
# and terms (words) on columns.  It will be a sparse matrix 
sms_dtm <- DocumentTermMatrix(corpus_clean)
sms_dtm
# we have 5559 rows(documents) and 7921 columns (terms) 
class(sms_dtm)

######### 6. split the data into a training dataset and test dataset
# doing sequential splitting (and not random)

sms_raw_train <- sms_data[1:4169,] # this split is from original data
sms_raw_test <- sms_data[4170:5559,]

sms_dtm_train <- sms_dtm[1:4169, ] # this split is from dtm matrix
sms_dtm_test <- sms_dtm[4170:5559, ]

sms_corpous_train <- corpus_clean[1:4169] # this split is from cleaned corpus
sms_corpous_test <- corpus_clean[4170:5559]

# sms_train_labels <- sms_raw[1:4169, ]$type
# sms_test_labels <- sms_raw[4170:5559, ]$type

### checking proportion of spam in original and split data
prop.table(table(sms_data$type))
prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))
# for original data, proportion of ham is 87% and spam is 13%
# even for both the training data and test data this proportion is same. 
# This suggests that the spam messages were divided evenly between the two datasets.

##### visualizing ham and spam raw-text-data using wordcloud

ham <- subset(sms_data, type=="ham") # 4812 records
spam <- subset(sms_data, type=="spam") #  747 records

### the ham cloud
wordcloud(ham$text, max.words = 100, scale = c(3,0.5))

# the spam cloud
wordcloud(spam$text, max.words = 50, scale = c(3,0.5),
          colors = brewer.pal(8,"Dark2"))


##### 8. creating indicator features for frequent words from sms_dtm_train
# eliminating any words that appear in less than 3 SMS messages
sms_freq_words  <- findFreqTerms(sms_dtm_train, 3)
str(sms_freq_words )
#  this has only 1949 words (out of 6578 words), that repeat atleast 3 times
sms_freq_words[1:5]
list(sms_freq_words[1:20])

### filtering DTM to include only these frequent words
sms_train <- DocumentTermMatrix(sms_corpous_train, list(dictionary=sms_freq_words))
# 4169 documents and 1949 terms
sms_test <- DocumentTermMatrix(sms_corpous_test, list(dictionary=sms_freq_words))
# 1390 documents and 1949 terms

########### 10. Changing feature-frequency to a categorical variable 
# this simply indicates yes or no depending on whether the word appears or not.
# In dtm matrix, the data is in numeric form. for naive bayes we need categorical
# type. so lets convert numeric to binary category. 0 if nor present and 1 if present
convert_counts <- function(x){
  x <- ifelse(x > 0, 1,0)
  x <- factor(x, levels = c(0,1), labels = c("No","Yes"))
}
class(sms_train)
sms_train <- apply(sms_train, MARGIN =2, convert_counts)
sms_test <- apply(sms_test, MARGIN =2, convert_counts)
str(sms_train)
4169*1949 # = 8125381 elements in sms_train
str(sms_test)
1390*1949 # = 2709110 elements in sms_test

View(sms_train)
# View(sms_test)

############## 11. Training a model on the data
# library(e1071)
# building the model on sms_train
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
sms_classifier$levels
sms_classifier[1]
#  ham 3605, spam 564
# sms_classifier[1:4]

####### 12. Evaluating model performance by using validation data
# using classifier to generate predictions on test data
sms_test_pred <- predict(sms_classifier, sms_test)
sms_test_pred[1:25]
sms_raw_test$type[1:25]
class(sms_test)

corpus_clean$content[1:10]

############## 13. Comparing the predictions to the true values

# library(gmodels)
table1 <- table(sms_test_pred, sms_raw_test$type)
table1
# sms_test_pred  ham spam
#          ham  1203   28
#          spam    4  155
CrossTable(sms_test_pred, sms_raw_test$type,
           prop.chisq=F, prop.c=F, prop.r=F,
           dnn=c("Predicted",'Actual'))
accuracy <- (sum(diag(table1))/ sum(table1))
accuracy # 0.9769

##  14. improving model performance by using laplace smoothing ----
# laplace smoother ensures that each feature has a nonzero probability
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
table2 <- table(sms_test_pred2, sms_raw_test$type)
table2
# sms_test_pred  ham spam
#          ham  1205   30
#          spam    2  153
(sum(diag(table2))/ sum(table2)) # 0.9769

# when use laplace=1, accuracy is 0.9769
# when use laplace=2 and 3, accuracy reduces to 0.9669 and 0.955 respectively

# though accuracy is same with or without laplace smoother, we see that
# in the latter case, only 2 ham sms are predicted as spam. without laplace
# smoother, 4 ham sms were predicted as spam. 


##################  lOGISTIC REGRESSION

# We are binding both the labels (outcome) and the predictors into one dataset

sms_train_labels <- sms_data[1:4169, ]$type
sms_train_labels <- as.data.frame(sms_train_labels)
class(sms_train_labels)
# sms_test_labels <- sms_raw[4170:5559, ]$type

sms_train_all <- as.data.frame(cbind(sms_train_labels, sms_train))
class(sms_train_all)
View(sms_train_all)

#### Running the logreg model
sms_logreg <- glm(sms_train_labels ~ . , data=sms_train_all, family="binomial")
summary(sms_logreg)
# Null deviance: 3304.431  on 4168  degrees of freedom
# Residual deviance:   18.705  on 2276  degrees of freedom
# AIC: 3804.7

# predicting for test data
sms_test_df <- as.data.frame(sms_test)
sms_logreg_pred <- predict(sms_logreg, sms_test_df)

# Confusion matrix and considering the threshold value as 0.5 
confusion<-table(sms_logreg_pred>0.5,sms_raw_test$type)
confusion
#        ham spam
# FALSE 1031   55
# TRUE   176  128

# Model Accuracy 
sum(diag(confusion)/sum(confusion)) # 0.8338129

# with Naive Bayes accuracy was 97%, but with logistic regression it is just 83%
# in NB only 2 ham sms were misclassified, in logreg 176 ham sms are misclassified as spam
# Also running the logreg model on sms-data is time consuming. 

'''
CONCLUSIONS:

We have used sms_data and built various models to classify if the sms is ham or
spam. Since we have to do binary classification, we have used Naive Bayes classifier
and Logistic Regression algorithm.

First we have cleaned the text to have only useful (meaningful) words.
We have built a document-term matrix. Then we have used Naive Bayes classifier
to train the model. It is then validated using the test data. We have used
laplace smoothing to improve our model.

We have also built another model using Logistic Regression algorithm. 

When we compare the 2 models, NB model gave an accuracy of 97% and
logistic regression model"s accuracy was only 83%. Whenever we have new sms data,
we can use the superior NB model to classify them into ham and spam.
'''



