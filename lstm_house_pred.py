# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam

max_features = 20000
maxlen = 800
ACTIVATION = 'relu'
EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE= 0.0001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
EPOCHS_DROP = 1
DROP = 0.1

train = pd.read_csv("features-training.csv")
target_train = pd.read_csv("target-training.csv")
test = pd.read_csv("features-validation.csv")
id_lb = test['id'].values.reshape((test['id'].values.shape[0], 1)).astype(int)
#train = train.sample(frac=1)

list_sentences_train = train["description"].fillna("CVxTz").values
y = target_train['price'].values
y = np.log1p(y)
list_sentences_test = test["description"].fillna("CVxTz").values

filename='lstm_house_pred'
print('tokenization...')
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
   # x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
   # x = Dropout(0.1)(x)
    x = Dense(1)(x)
    model = Model(inputs=inp, outputs=x)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss=root_mean_squared_error,
                  optimizer=adam,
                  metrics=['mean_squared_error'])

    return model

def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   if epoch % 10 == 0:
        model.save(filename + "_graph_" + str(epoch))
   return lrate

lrate = LearningRateScheduler(step_decay)

tbCallBack = TensorBoard(log_dir='./folder_' + filename, histogram_freq=0, write_graph=True, write_images=True)

model = get_model()
batch_size = 1024
epochs = 2


file_path="embeddings_source"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
model.load_weights(file_path)

callbacks_list = [checkpoint, early, lrate, tbCallBack] #early
#model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.03, callbacks=callbacks_list)

model.load_weights(file_path)

#features_train = model.predict(X_t, batch_size=2048)
#np.savetxt('features-embeddings/embbedings_source_features.csv', features_train, delimiter=',', fmt=['%.10f'])

y_lb = model.predict(X_te, batch_size=2048)

print (id_lb)
res = np.hstack([id_lb, y_lb])
np.savetxt('validation/embeddings_source_validation.csv', res, delimiter=',', fmt=['%d', '%.10f'])


