
#!pip install nltk
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

#!pip install seaborn

import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import numpy as np # linear algebra
import pandas as pd #data processing
import re

import requests
import seaborn as sns

train=pd.read_csv('train.csv')

print(train.isnull().sum())

train=train.fillna(' ')

train['content']=train['title']+' '+train['author']+train['text']

from nltk.corpus import stopwords
# corpus of nltk will hold the stopwords

stop=stopwords.words("english")

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = train['content']
Y_train = train['label']

#count_vectorizer = CountVectorizer(max_features=300)
#count_vectorizer.fit_transform(X_train)
#freq_term_matrix = count_vectorizer.transform(X_train)
#tfidf = TfidfTransformer(norm="l2")
#tfidf.fit(freq_term_matrix)
#tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)

tfidf = TfidfVectorizer(stop_words = 'english', max_df = 0.7, max_features = 1500)
# Fit and transform training set and transform test set
tfidf_train = tfidf.fit_transform(X_train) 
#tfidf_test = tfidf.transform(x_test)
#tfidf_test_final = tfidf.transform(test['text'])

tf_idf_matrix = tfidf_train

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, Y_train, random_state=0,test_size = 0.2)

from sklearn.metrics import classification_report,accuracy_score,f1_score,recall_score,precision_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test)
cr    = classification_report(y_test,pred)
print(cr)

print("test: ", accuracy_score(y_test,pred))

y_pred = pred

score = accuracy_score(y_test, y_pred)
f1score=f1_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')
print(f'F1 Score: {round(f1score * 100, 2)}%')
print(f'Recall: {round(recall * 100, 2)}%')
print(f'Precision: {round(precision * 100, 2)}%')

Accuracy = [95.84,96.15,96.42,96.42,96.56]
F1_Score=  [95.94,96.25,96.49,96.50,96.63]
Recall=    [96.78,96.93,97.11,97.21,97.54]
Precision= [95.12,95.32,95.73,95.89,96.06]
