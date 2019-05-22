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
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.initializers import *
from keras.constraints import *
from keras.regularizers import *
from keras.activations import *
from keras.optimizers import *
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


np.random.seed(9)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   return lrate

max_features = 20000
maxlen = 300
ACTIVATION = 'relu'
EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE= 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
EPOCHS_DROP = 1
DROP = 0.5

train = pd.read_csv("features-training.csv")
target_train = pd.read_csv("target-training.csv")
test = pd.read_csv("features-validation.csv")
id_lb = test['id'].values.reshape((test['id'].values.shape[0], 1)).astype(int)

list_sentences_train = train["description"].fillna("CVxTz").values
y = target_train['price'].values
y = np.log1p(y)
list_sentences_test = test["description"].fillna("CVxTz").values

filename='embeddings_source_attention_cnn'
print('tokenization...')
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

embed_size = 128
inp = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size)(inp)
x1 = SpatialDropout1D(0.0)(x)
att = Attention(maxlen)(x1)
# from benchmark kernel
x = Conv1D(128, 2, activation='relu', padding='same')(x1)
x = MaxPooling1D(5, padding='same')(x)
x = Conv1D(128, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(5, padding='same')(x)
x = Flatten()(x)
x = concatenate([x, att])
x = Dropout(0.1)(Dense(128, activation='relu') (x))
x = Dense(1)(x)

model = Model(inputs=inp, outputs=x)
adam = Adam(lr=LEARNING_RATE)
model.compile(loss=root_mean_squared_error,
              optimizer=adam,
              metrics=['mean_squared_error'])
model.summary()

batch_size = 1024
epochs = 15

lrate = LearningRateScheduler(step_decay)
filepath="best-models/embeddings_source_attention_cnn"
tbCallBack = TensorBoard(log_dir='./folder_' + filepath, histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, lrate]
model.load_weights(filepath)
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.03 ,callbacks = callbacks_list,verbose=1)

#Loading model weights
model.load_weights(filepath)

features_train = model.predict(X_t, batch_size=2048)
np.savetxt('features-embeddings/embeddings_source_attention_cnn_features.csv', features_train, delimiter=',', fmt=['%.10f'])

y_lb = model.predict(X_te, batch_size=2048)

print (id_lb)
res = np.hstack([id_lb, y_lb])
np.savetxt('validation/embeddings_source_attention_cnn_validation.csv', res, delimiter=',', fmt=['%d', '%.10f'])
