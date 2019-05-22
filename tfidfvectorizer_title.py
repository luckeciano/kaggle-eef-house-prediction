import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,  cross_val_score
from keras.optimizers import Adam
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Model, Sequential



LEARNING_RATE= 0.0001
EPOCHS_DROP = 1
DROP = 0.1
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
BATCH_SIZE = 1024
EPOCHS = 30
ACTIVATION = 'relu'

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   return lrate


train = pd.read_csv('features-training.csv')
y = pd.read_csv('target-training.csv')['price'].values
y = np.log1p(y)
test = pd.read_csv('features-test.csv')
val = pd.read_csv('features-validation.csv')
id_lb = test['id'].values.reshape((test['id'].values.shape[0], 1)).astype(int)
id_val = val['id'].values.reshape((val['id'].values.shape[0], 1)).astype(int)


train['title'] = train['title'].fillna('fillna')
test['title'] = test['title'].fillna('fillna')
val['title'] = val['title'].fillna('fillna')
X_train = train["title"].str.lower()
X_test = test["title"].str.lower()
X_val = val["title"].str.lower()

all_text = pd.concat([X_train, X_test])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_features = word_vectorizer.transform(X_train)
test_features = word_vectorizer.transform(X_test)
val_features = word_vectorizer.transform(X_val)

# char_vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='char',
#     ngram_range=(2, 6),
#     max_features=50000)
# char_vectorizer.fit(all_text)
# train_char_features = char_vectorizer.transform(X_train)
# test_char_features = char_vectorizer.transform(X_test)
# train_word_features = train_word_features.toarray()
#test_word_features = test_word_features.toarray()
# train_features = hstack([train_char_features, train_word_features]).toarray()
# test_features = hstack([test_char_features, test_word_features]).toarray()

def baseline_model():
  #Neural Network Design
  model =  Sequential()
  #model.add(Dense(64, input_dim = INPUT_SIZE, activation = ACTIVATION))
  model.add(Dense(512, input_dim=train_features.shape[1], activation = ACTIVATION))
  model.add(Dense(256, activation = ACTIVATION))
  model.add(Dense(128, activation = ACTIVATION))
  model.add(Dense(64, activation = ACTIVATION))
  model.add(Dense(32, activation = ACTIVATION))
  model.add(Dense(16, activation = ACTIVATION))
  # model.add(Dense(32, input_dim = INPUT_SIZE, activation = ACTIVATION))
  # model.add(Dense(32, input_dim = INPUT_SIZE, activation = ACTIVATION))
  model.add(Dense(1))

  adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
  model.compile(loss=root_mean_squared_error,
                optimizer=adam,
                metrics=['mean_squared_error'])
  model.load_weights('best-models/tfidfvectorizer_title')
  return model

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=1024, verbose=True)

#kfold = KFold(n_splits=10, random_state=9)
#results = cross_val_score(estimator, train_features, y, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(train_features, y,  validation_split = 0.03, verbose=True)
estimator.model.load_weights('tfidfvectorizer_title')
y_lb = np.expm1(estimator.predict(val_features)).reshape(id_val.shape)


print (id_val)
res = np.hstack([id_val, y_lb])
np.savetxt('validation/tfidfvectorizer_title_validation.csv', res, delimiter=',', fmt=['%d', '%.10f'])


# features = estimator.predict(train_features)
# np.savetxt('tfidfvectorizer_title_features.csv', features, delimiter=',', fmt=['%.10f'])
