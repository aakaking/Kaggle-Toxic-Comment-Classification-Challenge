# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:41:41 2018

@author: Anan
"""
import string
import pandas as pd 
import numpy as np
from collections import Counter
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

train = pd.read_csv('../clean_data/train_preprocessed.csv')

def clean_text(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

train['clean_text'] = train['comment_text'].apply(clean_text)
X_train = train['clean_text'].fillna("something").values

y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

toxic_comments = []
for i in range(len(train)):
    if 1 in y_train[i]:
        toxic_comments += train.iloc[i].clean_text
        
non_toxic_comments = []
for i in range(len(train)):
    if not (1 in y_train[i]):
        non_toxic_comments += train.iloc[i].clean_text
        
toxic_words = Counter(toxic_comments).most_common(6000)
non_toxic_words = Counter(non_toxic_comments).most_common(6000)

toxic = []
for element in toxic_words:
    toxic.append(element[0])
    
non_toxic = []
for element in non_toxic_words:
    non_toxic.append(element[0])
    
toxic_list = []
for word in toxic:
    if word not in non_toxic:
        toxic_list.append(word)
        
from gensim.models import KeyedVectors
en_model = KeyedVectors.load_word2vec_format('../crawl-300d-2M.vec')

toxic_correct = []
for word in toxic_list:
    if word in en_model.vocab:
        toxic_correct.append(word)
    else:
        b = TextBlob(word)
        b.correct()
        if b in en_model.vocab:
            toxic_correct.append(b)
        else: continue
    
import pickle
with open('toxic_words',"wb") as f:
    pickle.dump(toxic_correct,f)