#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install nltk')


# In[148]:


from urllib import request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test = test.set_index('id', drop = True)


# In[3]:


# Dropping all rows where text column is NaN
train.dropna(axis=0, how="any", thresh=None, subset=['text'], inplace=True)
test = test.fillna(' ')

# Checking length of each article
length = []
[length.append(len(str(text))) for text in train['text']]
train['length'] = length

# Removing outliers, it will reduce overfitting
train = train.drop(train['text'][train['length'] < 50].index, axis = 0)


# In[171]:


# Secluding labels in a new pandas dataframe for supervised learning
train_labels = train['label']
# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(train['text'], train_labels, test_size=0.2, random_state=0)

# Setting up Term Frequency - Inverse Document Frequency Vectorizer
#tfidf = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
tfidf = TfidfVectorizer(stop_words = 'english', max_features=1500)
# Fit and transform training set and transform test set
tfidf_train = tfidf.fit_transform(x_train) 
tfidf_test = tfidf.transform(x_test)
tfidf_test_final = tfidf.transform(test['text'])


# In[172]:


X_train = pd.DataFrame(tfidf_train.toarray())
X_test = pd.DataFrame(tfidf_test.toarray())


# In[173]:


clf = SVC(kernel = 'linear',tol = 0.01 )
model = clf.fit(X_train, y_train)


# In[174]:


y_train_pred= model.predict(X_train)
y_test_pred=model.predict(X_test)
train_accuracy = accuracy_score(y_train,y_train_pred)
test_accuracy = accuracy_score(y_test,y_test_pred)
print("Train : {}, Test: {}".format(train_accuracy,test_accuracy))


# In[175]:


f1score=f1_score(y_train, y_train_pred)
print(f1score)


# In[176]:


prec=precision_score(y_train, y_train_pred, average='weighted')
print(prec)


# In[177]:


recall=recall_score(y_train, y_train_pred, average='weighted')
print(recall)


# In[196]:


# Making Graphs
fig,ax=plt.subplots()
feature_x=[300,600,900,1200,1500]
accuracy_y_svm=[ 91.61,93.87,94.64,95.24,95.72]
accuracy_y_pac=[88.35,91.8,92.92,92.9,92.75]
accuracy_y_rf=[95.84,96.15,96.42,96.42,96.54]
# Function to plot
ax.plot(feature_x,accuracy_y_svm,marker=">")
plt.xlabel("Max Features")
plt.ylabel("Accuracy")
ax.plot(feature_x,accuracy_y_pac,marker="*")
ax.plot(feature_x,accuracy_y_rf,marker="^")
# function to show the plot
# plt_1 = plt.figure(figsize=(6, 3))
# for i, j in zip(feature_x, f1score_y1):
#    plt.text(i, j+0.5, '({}, {})'.format(i, j))
# for i, j in zip(feature_x, f1score_y2):
#    plt.text(i, j+0.5, '({}, {})'.format(i, j))
plt.legend(["SVM", "PAC","RF"])
# plt.show()
plt.savefig("feature_to_accuracy.png")


# In[197]:


# Making Graphs
fig,ax=plt.subplots()
feature_x=[300,600,900,1200,1500]
f1score_y_svm=[91.63022351797863,93.84587178390271,94.61048505634494,95.21068103870651,95.69055354625146]
f1score_y_pac=[88.96,91.86,92.76,92.91,92.62]
f1score_y_rf=[95.94,96.25,96.49,96.50,96.63]
# Function to plot
ax.plot(feature_x,f1score_y1_svm,marker=">")
plt.xlabel("Max Features")
plt.ylabel("F1 score")
ax.plot(feature_x,f1score_y2_pac,marker="*")
ax.plot(feature_x,f1score_y2_rf,marker="^")
# function to show the plot
# plt_1 = plt.figure(figsize=(6, 3))
# for i, j in zip(feature_x, f1score_y1):
#    plt.text(i, j+0.5, '({}, {})'.format(i, j))
# for i, j in zip(feature_x, f1score_y2):
#    plt.text(i, j+0.5, '({}, {})'.format(i, j))
plt.legend(["SVM", "PAC","RF"])
# plt.show()
plt.savefig("feature_to_f1.png")


# In[198]:


# Making Graphs
fig,ax=plt.subplots()
feature_x=[300,600,900,1200,1500]
precision_y_svm=[91.64,93.88,94.65,95.24,95.72]
precision_y_pac=[89.08,90.09,91.54,93.71,93.04]
precision_y_rf= [95.12,95.32,95.73,95.89,96.06]
# Function to plot
ax.plot(feature_x,precision_y_svm,marker=">")
plt.xlabel("Max Features")
plt.ylabel("precision")
ax.plot(feature_x,precision_y_pac,marker="*")
ax.plot(feature_x,precision_y_rf,marker="^")
# function to show the plot
# plt_1 = plt.figure(figsize=(6, 3))
# for i, j in zip(feature_x, f1score_y1):
#    plt.text(i, j+0.5, '({}, {})'.format(i, j))
# for i, j in zip(feature_x, f1score_y2):
#    plt.text(i, j+0.5, '({}, {})'.format(i, j))
plt.legend(["SVM", "PAC","RF"])
# plt.show()
plt.savefig("feature_to_precision.png")


# In[199]:


# Making Graphs
fig,ax=plt.subplots()
feature_x=[300,600,900,1200,1500]
recall_y_svm=[91.61,93.87,94.64,95.24,95.73]
recall_y_pac=[90.83,90.78,91.82,92.33,94.21]
recall_y_rf= [96.78,96.93,97.11,97.54,97.24]
# Function to plot
ax.plot(feature_x,recall_y_svm,marker=">")
plt.xlabel("Max Features")
plt.ylabel("Recall")
ax.plot(feature_x,recall_y_pac,marker="*")
ax.plot(feature_x,recall_y_rf,marker="^")
# function to show the plot
# plt_1 = plt.figure(figsize=(6, 3))
# for i, j in zip(feature_x, f1score_y1):
#    plt.text(i, j+0.5, '({}, {})'.format(i, j))
# for i, j in zip(feature_x, f1score_y2):
#    plt.text(i, j+0.5, '({}, {})'.format(i, j))
plt.legend(["SVM", "PAC","RF"])
# plt.show()
plt.savefig("feature_to_recall.png")


# In[200]:


C_Value=[0.1,0.25,0.5,0.7,1,1.5,2]
f1_score_y=[94.08,95.26,95.97,96.27,96.57,96.94,97.13]
plt.plot(C_Value,f1_score_y,marker=">",color="red")
plt.xlabel("C Paramater")
plt.ylabel("F1 score")
# # function to show the plot
# # plt.show()
# plt_1 = plt.figure(figsize=(10, 10))
for i, j in zip(C_Value, f1_score_y):
   plt.text(i, j+0.5, '({}, {})'.format(i, j))
plt.savefig("C_to_f1.png")

