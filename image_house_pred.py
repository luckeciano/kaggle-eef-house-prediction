# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.

from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from sklearn.metrics import mean_squared_error

ACTIVATION = 'relu'
EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE= 0.001
LOSS_FUNCTION = 'mean_squared_error'
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
EPOCHS_DROP = 10
DROP = 0.8
filename = 'image_house_pred'


train = pd.read_csv('train-images-features/train-images-features_0.csv', header=None)
for i in range(1, 20):
  print('Reading batch ' + str(i) + '...')
  train_i = pd.read_csv('train-images-features/train-images-features_' + str(i) + '.csv', header = None)
  train = pd.concat([train, train_i], axis = 0)
X_t = train.values[:, 1:]
print(train.values[:, 1:].shape)

target_train = pd.read_csv("target-training.csv")
y = target_train['price'].values
y = np.log1p(y)
#train = train.sample(frac=1)

test = pd.read_csv('validation-image-features/validation-images-features_0.csv', header=None)
for i in range(1, 20):
  print('Reading batch ' + str(i) + '...')
  test_i = pd.read_csv('validation-image-features/validation-images-features_' + str(i) + '.csv', header = None)
  test = pd.concat([test, test_i], axis = 0)
X_te = test.values[:, 1:]
print(test.values[:, 1:].shape)


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

model =  Sequential() 
model.add(Dense(256,  input_dim=264,  activation = ACTIVATION))
model.add(Dense(128,  activation = ACTIVATION))
model.add(Dense(64,  activation = ACTIVATION))
model.add(Dense(1))
model.load_weights('image_house_pred_graph_90')

adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
model.compile (loss = root_mean_squared_error, optimizer = adam, metrics = ['mse', root_mean_squared_error])

lrate = LearningRateScheduler(step_decay)
tbCallBack = TensorBoard(log_dir='./folder_' + filename, histogram_freq=0, write_graph=True, write_images=True)


file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early, lrate, tbCallBack] #early
#model.fit(X_t, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.03, verbose = 1, callbacks=callbacks_list)
print("Predicting Features...")

#y_hat = model.predict(X_t, batch_size=2048)

#np.savetxt('feature-embeddings/image-features.csv', y_hat, delimiter=',', fmt=['%.10f'])

print("Predicting LB...")
y_hat = model.predict(X_te, batch_size=2048)

np.savetxt('validation/image-validation.csv', y_hat, delimiter=',', fmt=['%.10f'])
