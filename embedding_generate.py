import numpy as np

import pandas as pd

from keras.preprocessing import text, sequence


import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

import pickle
with open('../embedding_matrix.txt','rb') as f:
    embedding_matrix = pickle.load(f)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


max_features = 30000
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

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if word not in en_model.vocab: continue
    if i >= max_features: continue
    embedding_vector = en_model[word]
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

import pickle
with open('embedding_matrix_',"wb") as f:
    pickle.dump(embedding_matrix,f)