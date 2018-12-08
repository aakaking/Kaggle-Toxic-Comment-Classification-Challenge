# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 09:56:07 2018

@author: Anan
"""

import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

from textblob import TextBlob
import editdistance
from nltk.tokenize import word_tokenize
import string


import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

def clean_text(text):
    if hasattr(text, "decode"):
        text = text.decode("utf-8")
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words



train = pd.read_csv('clean_data/train_preprocessed.csv')
test = pd.read_csv('raw_data/test.csv')
submission = pd.read_csv('../sample_submission.csv')

train['clean_text'] = train['comment_text'].apply(clean_text)
X_train = train['clean_text'].fillna("something").values
test['clean_text'] = test['comment_text'].apply(clean_text)
X_test = test['clean_text'].fillna("something").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

max_features = 100000
maxlen = 400
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

from gensim.models import KeyedVectors
en_model = KeyedVectors.load_word2vec_format('../crawl-300d-2M.vec')

import pickle
with open('toxic_words','br') as f:
    toxic_words = pickle.load(f)
    
def correct1(word):
    if len(word) < 3:
        return word
    distance = []
    for word in toxic_words:
        distance.append(editdistance.eval('pedopiles',word))
    if min(distance) < 2:
        index = distance.index(min(distance))
        return toxic_words[index]
    return word
def correct2(word):
    b = TextBlob('whateva')
    return b.correct()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    if word not in en_model.vocab: 
        word = correct1(word)
        if word not in en_model.vocab:
            word = correct2(word)
            if word not in en_model.vocab:
                continue
    embedding_vector = en_model[word]
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector