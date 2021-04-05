#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:29:54 2021

@author: patkar
"""

import nltk
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords  

data = pd.read_csv('/home/patkar/nlp/nlpfiles/sms_spam_classification/SMSSpamCollection',
                   sep = '\t', names = ['label', 'message'])

stemmer = PorterStemmer()
corpus = []

for i in range(0, len(data)):
    text = re.sub('[^a-zA-Z]', ' ', data['message'][i])
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)

# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)

# Independent data
x = cv.fit_transform(corpus).toarray()
# Dependent data
y = pd.get_dummies(data['label'])
y = y.iloc[:,1].values

# Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,
                                                    random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(x_train, y_train)

# Preditction
y_pred = model.predict(x_test)

# Confusion matrix, accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)

print(f'Confusion matrix: \n{confusion_matrix} \n \nAccuracy score : \n{accuracy_score}')