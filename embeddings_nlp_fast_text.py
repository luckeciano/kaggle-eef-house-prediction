# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
np.random.seed(9)
LEARNING_RATE= 0.001
EPOCHS_DROP = 1
DROP = 0.1

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   return lrate


EMBEDDING_FILE = 'nlp-embeddings/fasttext_skip_s600.txt'
train = pd.read_csv('features-training.csv')
y = pd.read_csv('target-training.csv')['price'].values
y = np.log1p(y)
test = pd.read_csv('features-test.csv')
val = pd.read_csv('features-validation.csv')
id_lb = test['id'].values.reshape((test['id'].values.shape[0], 1)).astype(int)
id_val = val['id'].values.reshape((val['id'].values.shape[0], 1)).astype(int)

train['description'] = train['description'].fillna('fillna')
test['description'] = test['description'].fillna('fillna')
val['description'] = val['description'].fillna('fillna')
X_train = train["description"].str.lower()
X_test = test["description"].str.lower()
X_val = val["description"].str.lower()

max_features=100000
maxlen=800
embed_size=600

tok=text.Tokenizer(num_words=max_features,lower=True)
tok.fit_on_texts(list(X_train)+list(X_test))
X_train=tok.texts_to_sequences(X_train)
X_test=tok.texts_to_sequences(X_test)
X_val = tok.texts_to_sequences(X_val)
X_train=sequence.pad_sequences(X_train,maxlen=maxlen)
X_test=sequence.pad_sequences(X_test,maxlen=maxlen)
X_val = sequence.pad_sequences(X_val, maxlen=maxlen)

embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tok.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

sequence_input = Input(shape=(maxlen, ))
x = Embedding(num_words, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
x = Bidirectional(LSTM(50, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dense(1)(x)
model = Model(inputs=sequence_input, outputs=x)
adam = Adam(lr=LEARNING_RATE)
model.compile(loss=root_mean_squared_error,
              optimizer=adam,
              metrics=['mean_squared_error'])
model.summary()

batch_size = 1024
epochs = 3

lrate = LearningRateScheduler(step_decay)
filepath="best-models/fast_text_best"
tbCallBack = TensorBoard(log_dir='./folder_' + filepath, histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, lrate, tbCallBack]
model.load_weights(filepath)
model.fit(X_train, y, batch_size=batch_size, epochs=epochs, validation_split=0.03 ,callbacks = callbacks_list,verbose=1)
#Loading model weights

model.load_weights(filepath)

features_train = model.predict(X_train, batch_size=2048)
np.savetxt('features-embeddings/embbedings_fast_text_features.csv', features_train, delimiter=',', fmt=['%.10f'])

y_lb = model.predict(X_val, batch_size=2048)

print (id_val)
res = np.hstack([id_val, y_lb])
np.savetxt('validation/embeddings_fasttext_validation.csv', res, delimiter=',', fmt=['%d', '%.10f'])
