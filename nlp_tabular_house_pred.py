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
	#train['lon'] = train.groupby("state_name")["lon"].transform(
	#    lambda x: x.fillna(x.mean()))   
	#train['lon'] = train['lon'].fillna(train['lon'].mean())
	#train['lat'] = train.groupby("state_name")["lat"].transform(
	#    lambda x: x.fillna(x.mean()))        
	#train['lat'] = train['lat'].fillna(train['lat'].mean())
	train['rooms'] = train['rooms'].fillna(train['rooms'].mean())
	train['surface_covered_in_m2'] = train['surface_covered_in_m2'].fillna(train['surface_covered_in_m2'].mean())
	train['surface_total_in_m2'] = train['surface_total_in_m2'].fillna(train['surface_covered_in_m2'])

	

	return train

def fit_transform_outliers(train):
	cols = ['floor', 'expenses','surface_covered_in_m2', 'surface_total_in_m2']
	qs = []
	for col in cols:
	    q = train[col].quantile(0.999)
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
nlp = pd.read_csv('feature-nlp.csv')
values_from_title = pd.read_csv('feature_title.csv')
values_from_title['values'] = np.log1p(values_from_title['values'])
train = pd.concat([train, nlp['nlp'], values_from_title['values']], axis=1)
print(nlp.head())

test['price'] = np.log1p(test['price'])
train = process_data(train)
train, qs = fit_transform_outliers(train)

train, scaler = scale_data(train)


train = pd.concat([train,test['price']], axis=1)
print(train.head())

y = train['price']
del train['price']
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

filename = 'house_prediction'
model =  Sequential()

ACTIVATION = 'relu'
EPOCHS = 12
BATCH_SIZE = 1024
LEARNING_RATE= 0.01
LOSS_FUNCTION = 'mean_squared_error'
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
EPOCHS_DROP = 3
DROP = 0.1
model.add(Dense(512,  input_dim=15,  activation = ACTIVATION))
model.add(Dense(256, activation = ACTIVATION))
model.add(Dense(128,  activation = ACTIVATION))
model.add(Dense(64,  activation = ACTIVATION))
model.add(Dense(32,  activation = ACTIVATION))
model.add(Dense(1))

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

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
tbCallBack = TensorBoard(log_dir='./' + filename, histogram_freq=0, write_graph=True, write_images=True)
filepath="nlp_tabular_best"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
#rmsprop = RMSprop(lr=LEARNING_RATE)
model.compile (loss = root_mean_squared_error, optimizer = adam, metrics = ['mse', root_mean_squared_error])
model.fit (X, y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1, callbacks=[lrate, tbCallBack, checkpoint], validation_split=0.1)
model.load_weights(filepath)
y_hat = model.predict(X)
print ("Total LMSE: " + str(mean_squared_error(y_hat, y)))

#y_hat = np.expm1(y_hat)
#np.savetxt('y_hat.csv', y_hat, delimiter=',', fmt=['%.10f'])

lb = pd.read_csv('features-test.csv')
nlp_test = pd.read_csv('nlp-test.csv')
values_from_title_test = pd.read_csv('title-test.csv')
values_from_title_test['values'] = np.log1p(values_from_title_test['values'])
lb = pd.concat([lb, nlp_test['nlp'], values_from_title_test['values']], axis=1)
id_lb = lb['id'].values.reshape((lb['id'].values.shape[0], 1)).astype(int)
lb = process_data(lb)
lb = transform_outliers(lb, qs)
lb = pd.DataFrame(scaler.transform(lb))
y_lb = np.expm1(model.predict(lb.values))

print (id_lb)
res = np.hstack([id_lb, y_lb])
np.savetxt('lb.csv', res, delimiter=',', fmt=['%d', '%.10f'])
