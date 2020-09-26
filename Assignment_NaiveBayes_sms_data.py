######### ASSIGNMENT NAIVE BAYES model using spam_ham dataset 

# BUSINESS PROBLEM: Classifying the emails as spam or ham using Naive Bayes model

# importing libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# loading data
email_data = pd.read_csv("E:/NaiveBayes/ham_spam.csv", encoding='ISO-8859-1')

### understanding the dataset
email_data.columns # ['type', 'text']
email_data.shape # 5559 records, 2 columns
email_data.dtypes # object (not numeric)
email_data.ndim
# email_data.size
# email_data.groupby('type').mean() # if had numeric variables, would make sense

email_data.describe()

email_data.head()

email_data.type.value_counts()
# ham     4812
# spam     747

###################### cleaning data
# The email text has numbers, punctuations that are not useful in classifying.
# hence we will be removing them here. Also converting remaining words to lower
# case 

import re

# importing stop words
''' To remove stop words, not used here.
stop_words = []
with open('E:/Datasets/stop.txt') as stop:
    stop_words = stop.read() 
    
# since all words are treated as one sentence, we need to split each word
stop_words = stop_words.split('\n')  '''

def cleaning_text(i):
    i = re.sub('[^A-Za-z" "]+'," ",i)
    i = re.sub('[0-9]+'," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>5:
            w.append(word)
    return (" ").join(w)
# example
cleaning_text("This is Awesome 1231312 $#%$# am i the queen and princess")

email_data.text = email_data.text.apply(cleaning_text)

email_data.shape
email_data = email_data.loc[email_data.text !=" ",:]


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

email_train,email_test = train_test_split(email_data, test_size=0.3, random_state=123)

## #### preparing term frequency matrix
def split_words(i):
    return i.split(" ")

email_bow = CountVectorizer(analyzer=split_words).fit(email_data.text)

all_matrix = email_bow.transform(email_data.text)
all_matrix.shape # 5559, 4867. we have 4867 terms 

train_matrix = email_bow.transform(email_train.text)
train_matrix.shape # (3891, 4867)

test_matrix = email_bow.transform(email_test.text)
test_matrix.shape # (1668, 4867)

################# Preparing a naive bayes model using TERM FREQUENCY matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB

####### Multinomial Naive Bayes
model_mb = MultinomialNB()
model_mb.fit(train_matrix, email_train.type)

train_pred_mb = model_mb.predict(train_matrix)
np.mean(train_pred_mb == email_train.type) # 0.983

test_pred_mb = model_mb.predict(test_matrix)
np.mean(test_pred_mb == email_test.type) # 0.952

pd.crosstab(train_pred_mb, email_train.type) # 8 ham classified as spam
pd.crosstab(test_pred_mb, email_test.type) # 50 ham predicted as spam

####### Gaussian Naive Bayes
model_gb = GaussianNB()
model_gb.fit(train_matrix.toarray(), email_train.type.values )

train_pred_gb = model_gb.predict(train_matrix.toarray())
np.mean(train_pred_gb == email_train.type) # 0.757

test_pred_gb = model_gb.predict(test_matrix.toarray())
np.mean(test_pred_gb == email_test.type) # 0.611

# since Gaussian NB model is for numeric data and our email-data is of text type,
#  the model does not fit properly. Hence, we are getting low accuracy

########### Preparing a naive bayes model using tfidf matrix

# Above  modelling was done on only term frequencies tf.
# next lets do on tfidf (term frq inverse doc frq)

tfidf_transformer = TfidfTransformer().fit(all_matrix)

train_tfidf = tfidf_transformer.transform(train_matrix)
train_tfidf.shape # (3891, 4867)
test_tfidf = tfidf_transformer.transform(test_matrix)
test_tfidf.shape # (1668, 4867)

# Multinomial Naive Bayes
model_tfidf_mb = MultinomialNB()
model_tfidf_mb.fit(train_tfidf, email_train.type)

train_tfidf_pred_mb = model_tfidf_mb.predict(train_tfidf)
np.mean(train_tfidf_pred_mb == email_train.type) # 0.959

test_tfidf_pred_mb = model_tfidf_mb.predict(test_tfidf)
np.mean(test_tfidf_pred_mb == email_test.type) # 0.941

email_train.type.value_counts()
''' ham     3376
    spam     515 '''
 
# finding error per category of ham-spam using tfidf
pd.crosstab(train_tfidf_pred_mb ,email_train.type)
# in train, no ham has been predicted as spam, (no alpha error), which is good
pd.crosstab(test_tfidf_pred_mb ,email_test.type)
# in test, 5 ham mails have been predicted as spam, this error has to be minimized

# finding error per category of ham-spam using tf
pd.crosstab(train_pred_mb, email_train.type) # 8 ham wrongly categorised as spam
pd.crosstab(test_pred_mb, email_test.type) # 50 wrongly classified as spam

# when we use tfidf, accuracy is less than tf matrix. 
# But alpha error is very less. For this business problem, if ham mail gets
# predicted as spam , then this is problematic. 
# So, for this dataset, Multinomial NB using tfidf matrix has given better results.

'''
CONCLUSIONS:

We have used sms_data and built various models to classify if the sms is ham or
spam. We have used Naive Bayes classifier. We have built models using only tf (term
freqency) and also used tfidf (term frequency inverse document frequency).

Multinomial naive bayes model has given better results.

'''

  

