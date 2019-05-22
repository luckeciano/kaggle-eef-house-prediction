#loading need libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def process_data(train):

	categories = ['property_id', 'currency', 'property_type', 'place_name', 'state_name']
	for cat in categories:
	    train[cat] = pd.Categorical(train[cat], categories=train[cat].unique()).codes

	train['created_on_year'] = pd.DatetimeIndex(train['created_on']).year
	train['created_on_month'] = pd.DatetimeIndex(train['created_on']).month
	train['created_on_day'] = pd.DatetimeIndex(train['created_on']).day

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
test = pd.read_csv('target-training.csv')
features_train_names = ['embeddings_source_300_features.csv', 'embeddings_source_title_features.csv' , 'feature-target-train.csv', 'image-features.csv']


features_test_names = [ 'embeddings_source_300_lb.csv', 'embeddings_source_title_lb.csv', 'feature-target-test.csv', 'image-lb.csv']

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


test['price'] = np.log1p(test['price'])
train = process_data(train)
train, qs = fit_transform_outliers(train)
train = pd.concat([train, features_train], axis=1)
train, scaler = scale_data(train)


y = test['price']
X = train.values
y = y.values

from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model, Sequential
from keras.layers import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error

filename = 'final_model'
model =  Sequential()

ACTIVATION = 'relu'
EPOCHS =   20
BATCH_SIZE = 1024
LEARNING_RATE= 0.01
LOSS_FUNCTION = 'root_mean_squared_error'
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
EPOCHS_DROP = 4
DROP = 0.5
model.add(Dense(512,  input_dim=17,  activation = ACTIVATION))
model.add(Dense(256, activation = ACTIVATION))
model.add(Dense(128,  activation = ACTIVATION))
model.add(Dense(64,  activation = ACTIVATION))
model.add(Dense(1))
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

model.compile (loss = root_mean_squared_error, optimizer = adam, metrics = ['mse', root_mean_squared_error])
model.fit (X, y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1, callbacks=[lrate, tbCallBack, checkpoint], validation_split=0.03)
model.load_weights(filename)

lb = pd.read_csv('features-test.csv')
id_lb = lb['id'].values.reshape((lb['id'].values.shape[0], 1)).astype(int)
lb = process_data(lb)
lb = transform_outliers(lb, qs)
lb = pd.concat([lb, features_test], axis=1)
lb = pd.DataFrame(scaler.transform(lb))
y_lb = np.expm1(model.predict(lb.values))

print (id_lb)
res = np.hstack([id_lb, y_lb])
np.savetxt('lb.csv', res, delimiter=',', fmt=['%d', '%.10f'])
