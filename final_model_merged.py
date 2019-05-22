#loading need libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import text, sequence
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def process_data(train):

	a = train['place_with_parent_names'].str.split('|')
	train['city_name'] = list(zip(*a))[3]

	categories = ['property_id', 'currency', 'property_type', 'place_name', 'state_name', 'city_name']
	for cat in categories:
	    train[cat] = pd.Categorical(train[cat], categories=train[cat].unique()).codes

	train['created_on_year'] = pd.DatetimeIndex(train['created_on']).year
	train['created_on_month'] = pd.DatetimeIndex(train['created_on']).month
	train['created_on_day'] = pd.DatetimeIndex(train['created_on']).day
	train['collected_on_month'] = pd.DatetimeIndex(train['collected_on']).month
	train['collected_on_year'] = pd.DatetimeIndex(train['collected_on']).year

	#Correlation between train attributes

	#Separate variable into new dataframe from original dataframe which has only numerical values
	train_corr = train.select_dtypes(include=[np.number])
	del train_corr['id']
	del train_corr['geonames_id']
	del train_corr['lat']
	del train_corr['lon']
	train = train_corr

	train['floor'] = train['floor'].fillna(0) #high missing ratio
	train['expenses'] = train['expenses'].fillna(train['expenses'].median())
	train['rooms'] = train['rooms'].fillna(train['rooms'].mean())
	train['surface_covered_in_m2'] = train['surface_covered_in_m2'].fillna(train['surface_covered_in_m2'].mean())
	train['surface_total_in_m2'] = train['surface_total_in_m2'].fillna(train['surface_covered_in_m2'])

	

	return train

def fit_transform_outliers(train):
	cols = ['floor', 'expenses','surface_covered_in_m2', 'surface_total_in_m2']
	qs = []
	for col in cols:
	    q = train[col].quantile(0.99)
	    qs.append(q)
	    train.loc[train[col] > q, col] = q
	print(qs)
	return train,qs

def transform_outliers(train,qs):
	cols = ['floor', 'expenses','surface_covered_in_m2', 'surface_total_in_m2']
	for col, q in zip(cols, qs):
	    train.loc[train[col] > q, col] = q
	print(qs)
	return train


def scale_data(train):
	scaler = StandardScaler()
	train = scaler.fit_transform(train)
	return pd.DataFrame(train), scaler


train = pd.read_csv('features-training.csv')
test = pd.read_csv('features-test.csv')


max_features = 20000
list_sentences_train = train["description"].fillna("CVxTz").values
list_sentences_test = test["description"].fillna("CVxTz").values

print('tokenization description...')
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t_description = sequence.pad_sequences(list_tokenized_train, maxlen=300)
X_te_description = sequence.pad_sequences(list_tokenized_test, maxlen=300)


list_sentences_train = train["title"].fillna("CVxTz").values
list_sentences_test = test["title"].fillna("CVxTz").values


print('tokenization title...')
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t_title = sequence.pad_sequences(list_tokenized_train, maxlen=150)
X_te_title = sequence.pad_sequences(list_tokenized_test, maxlen=150)



features_train_names = ['feature-target-train.csv', 'image-features.csv']

features_test_names = ['feature-target-test.csv', 'image-lb.csv']

needs_to_transform = ['feature-target-train.csv', 'feature-target-test.csv', 'tfidfvectorizer_lb.csv', 'tfidfvectorizer_title_lb.csv']

dataframes = []
for f in features_train_names:
	print(f)
	t = pd.read_csv('feature-embeddings/' + f, names=['feature'], header=None)['feature']
	if f in needs_to_transform:
		print("Need to log: ")
		print(t.head())
		t = np.log1p(t)
	print(t.head())
	dataframes.append(t)

features_train = pd.concat(dataframes, axis=1)

dataframes = []
for f in features_test_names:
	print(f)
	if f in ['image-lb.csv', 'feature-target-test.csv']:
		t = pd.read_csv('feature-embeddings/' + f, names=['feature'], header=None)['feature']
	else:
		t = pd.read_csv('feature-embeddings/' + f, names=['id', 'feature'], header=None)['feature']
	if f in needs_to_transform:
		print("Need to log: ")
		t = np.log1p(t)
	print(t.head())
	dataframes.append(t)
features_test = pd.concat(dataframes, axis=1)

test = pd.read_csv('target-training.csv')
test['price'] = np.log1p(test['price'])
train = process_data(train)
train, qs = fit_transform_outliers(train)
train = pd.concat([train, features_train], axis=1)
train, scaler = scale_data(train)



y = test['price']
X = train.values
Y = y.values

filename = 'final_model_merged'

ACTIVATION = 'relu'
EPOCHS =   3
BATCH_SIZE = 2048
LEARNING_RATE= 0.01
LOSS_FUNCTION = 'root_mean_squared_error'
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
EPOCHS_DROP = 2
DROP = 0.1

embed_size = 128
inp = Input(shape=(300, ))
emb = Embedding(max_features, embed_size)
x = emb(inp)
x = Bidirectional(LSTM(50, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)

inp2 = Input(shape=(150, ))
y = emb(inp2)
y = Bidirectional(LSTM(50, return_sequences=True))(y)
y = GlobalMaxPool1D()(y)
y = Dense(50, activation="relu")(y)

inp_tabular = Input(shape = (train.shape[1], ))
w = Dense(50, activation="relu")(inp_tabular)

z = Concatenate()([x, y, w])
z = Dense(512, activation="relu")(z)
z = Dropout(0.1)(z)
z = Dense(128, activation="relu")(z)
z = Dropout(0.1)(z)
z = Dense(1)(z)


model = Model(inputs=[inp, inp2, inp_tabular], outputs=z)
model.summary()
adam = Adam(lr=LEARNING_RATE)
model.compile(loss=root_mean_squared_error,
              optimizer=adam,
              metrics=['mean_squared_error'])
#model.load_weights(filename)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   return lrate

lrate = LearningRateScheduler(step_decay)
tbCallBack = TensorBoard(log_dir='./folder_' + filename, histogram_freq=0, write_graph=True, write_images=True)

adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

print(X_t_description.shape, X_t_title.shape, train[:10].shape, y.shape)

model.compile (loss = root_mean_squared_error, optimizer = adam, metrics = ['mse', root_mean_squared_error])
x = 0
model.fit (x = [X_t_description, X_t_title, train], y = Y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1, callbacks=[lrate, tbCallBack, checkpoint], validation_split=0.03)
model.load_weights(filename)

lb = pd.read_csv('features-test.csv')
id_lb = lb['id'].values.reshape((lb['id'].values.shape[0], 1)).astype(int)
lb = process_data(lb)
lb = transform_outliers(lb, qs)
lb = pd.concat([lb, features_test], axis=1)
lb = pd.DataFrame(scaler.transform(lb))

ft = np.expm1(model.predict([X_t_description, X_t_title, train]))
y_lb = np.expm1(model.predict([X_te_description, X_te_title, lb]))

np.savetxt('features-final-merged.csv', ft)

print (id_lb)
res = np.hstack([id_lb, y_lb])
np.savetxt('lb_final_merged.csv', res, delimiter=',', fmt=['%d', '%.10f'])
