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

# https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

class Capsule(Layer):
    """Capsule Layer implementation in Keras
       This implementation is based on Dynamic Routing of Capsules,
       Geoffrey Hinton et. al.
       The Capsule Layer is a Neural Network Layer which helps
       modeling relationships in image and sequential data better
       than just CNNs or RNNs. It achieves this by understanding
       the spatial relationships between objects (in images)
       or words (in text) by encoding additional information
       about the image or text, such as angle of rotation,
       thickness and brightness, relative proportions etc.
       This layer can be used instead of pooling layers to
       lower dimensions and still capture important information
       about the relationships and structures within the data.
       A normal pooling layer would lose a lot of
       this information.
       This layer can be used on the output of any layer
       which has a 3-D output (including batch_size). For example,
       in image classification, it can be used on the output of a
       Conv2D layer for Computer Vision applications. Also,
       it can be used on the output of a GRU or LSTM Layer
       (Bidirectional or Unidirectional) for NLP applications.
       The default activation function is 'linear'. But, this layer
       is generally used with the 'squash' activation function
       (recommended). To use the squash activation function, do :
       from keras_contrib.activations import squash
       capsule = Capsule(num_capsule=10,
                         dim_capsule=10,
                         routings=3,
                         share_weights=True,
                         activation=squash)
       # Example usage :
           1). COMPUTER VISION
           input_image = Input(shape=(None, None, 3))
           conv_2d = Conv2D(64,
                            (3, 3),
                            activation='relu')(input_image)
           capsule = Capsule(num_capsule=10,
                             dim_capsule=16,
                             routings=3,
                             activation='relu',
                             share_weights=True)(conv_2d)
           2). NLP
           maxlen = 72
           max_features = 120000
           input_text = Input(shape=(maxlen,))
           embedding = Embedding(max_features,
                                 embed_size,
                                 weights=[embedding_matrix],
                                 trainable=False)(input_text)
           bi_gru = Bidirectional(GRU(64,
                                      return_seqeunces=True))(embedding)
           capsule = Capsule(num_capsule=5,
                             dim_capsule=5,
                             routings=4,
                             activation='sigmoid',
                             share_weights=True)(bi_gru)
       # Arguments
           num_capsule : Number of Capsules (int)
           dim_capsules : Dimensions of the vector output of each Capsule (int)
           routings : Number of dynamic routings in the Capsule Layer (int)
           share_weights : Whether to share weights between Capsules or not
           (boolean)
           activation : Activation function for the Capsules
           regularizer : Regularizer for the weights of the Capsules
           initializer : Initializer for the weights of the Caspules
           constraint : Constraint for the weights of the Capsules
       # Input shape
            3D tensor with shape:
            (batch_size, input_num_capsule, input_dim_capsule)
            [any 3-D Tensor with the first dimension as batch_size]
       # Output shape
            3D tensor with shape:
            (batch_size, num_capsule, dim_capsule)
       # References
        - [Dynamic-Routing-Between-Capsules]
          (https://arxiv.org/pdf/1710.09829.pdf)
        - [Keras-Examples-CIFAR10-CNN-Capsule]"""

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 initializer='glorot_uniform',
                 activation=None,
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights

        self.activation = activations.get(activation)
        self.regularizer = regularizers.get(regularizer)
        self.initializer = initializers.get(initializer)
        self.constraint = constraints.get(constraint)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1,
                                            input_dim_capsule,
                                            self.num_capsule *
                                            self.dim_capsule),
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,
                                     constraint=self.constraint,
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule *
                                            self.dim_capsule),
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,
                                     constraint=self.constraint,
                                     trainable=True)

        self.build = True

    def call(self, inputs):
        if self.share_weights:
            u_hat_vectors = K.conv1d(inputs, self.W)
        else:
            u_hat_vectors = K.local_conv1d(inputs, self.W, [1], [1])

        # u_hat_vectors : The spatially transformed input vectors (with local_conv_1d)

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        u_hat_vectors = K.reshape(u_hat_vectors, (batch_size,
                                                  input_num_capsule,
                                                  self.num_capsule,
                                                  self.dim_capsule))

        u_hat_vectors = K.permute_dimensions(u_hat_vectors, (0, 2, 1, 3))
        routing_weights = K.zeros_like(u_hat_vectors[:, :, :, 0])

        for i in range(self.routings):
            capsule_weights = K.softmax(routing_weights)
            outputs = K.batch_dot(capsule_weights, u_hat_vectors, [2, 2])
            if K.ndim(outputs) == 4:
                outputs = K.sum(outputs, axis=1)
            if i < self.routings - 1:
                outputs = K.l2_normalize(outputs, -1)
                routing_weights = K.batch_dot(outputs, u_hat_vectors, [2, 3])
                if K.ndim(routing_weights) == 4:
                    routing_weights = K.sum(routing_weights, axis=1)

        return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = {'num_capsule': self.num_capsule,
                  'dim_capsule': self.dim_capsule,
                  'routings': self.routings,
                  'share_weights': self.share_weights,
                  'activation': activations.serialize(self.activation),
                  'regularizer': regularizers.serialize(self.regularizer),
                  'initializer': initializers.serialize(self.initializer),
                  'constraint': constraints.serialize(self.constraint)}

        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(x, axis=-1):
    """
    Squash activation function (generally used in Capsule layers).
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x



LEARNING_RATE= 0.01
EPOCHS_DROP = 1
DROP = 0.8

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


EMBEDDING_FILE = 'nlp-embeddings/wang2vec_skip_s600.txt'
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

max_features=100000
maxlen=150
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
x1 = SpatialDropout1D(0.3)(x)
bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(x1)
max_pool_lstm = GlobalMaxPooling1D()(bi_lstm)
attention_lstm = AttentionWeightedAverage()(bi_lstm)
capsule = Capsule(num_capsule=5, dim_capsule=5, routings=4, activation=squash)(bi_lstm)
capsule = Flatten()(capsule)
x = concatenate([max_pool_lstm, attention_lstm, capsule], axis=1)
x = Dense(1)(x)

model = Model(inputs=sequence_input, outputs=x)
adam = Adam(lr=LEARNING_RATE)
model.compile(loss=root_mean_squared_error,
              optimizer=adam,
              metrics=['mean_squared_error'])
model.summary()

batch_size = 1024
epochs = 4

lrate = LearningRateScheduler(step_decay)
filepath="best-models/wang2vec_attention_capsule_title"
tbCallBack = TensorBoard(log_dir='./folder_' + filepath, histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, lrate]
model.load_weights(filepath)
#model.fit(X_train, y, batch_size=batch_size, epochs=epochs, validation_split=0.03 ,callbacks = callbacks_list,verbose=1)

#Loading model weights
model.load_weights(filepath)

# features_train = model.predict(X_train, batch_size=2048)
# np.savetxt('features-embeddings/wang2vec_attention_capsule_title_features.csv', features_train, delimiter=',', fmt=['%.10f'])

y_lb = model.predict(X_val, batch_size=2048)

print (id_val)
res = np.hstack([id_val, y_lb])
np.savetxt('validation/wang2vec_attention_capsule_title_validation.csv', res, delimiter=',', fmt=['%d', '%.10f'])

