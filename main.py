from functions import *
from create_raw_data import *
import sklearn.preprocessing

import pandas as pd

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)



df = create_finance(returns=True)

data = prepare_data(df,20,20,10)

data.create_windows(20)


#Not for LSTM
x_train = data.x_train.reshape(len(data.x_train),-1)
x_valid = data.x_valid.reshape(len(data.x_valid),-1)
x_test = data.x_test.reshape(len(data.x_test),-1)

svr = SVR(C=100,verbose=True)

svr.fit(x_train,data.y_train.ravel())

print(svr.score(x_test,data.y_test.ravel()))


i = Input(shape=(20, 14))
x = LSTM(20)(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(lr=0.1),
)

# train the RNN
r = model.fit(
  data.x_train, data.y_train.ravel(),
  epochs=80,
  validation_data=(data.x_valid,data.y_valid.ravel()),
)




plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.figure()
outputs = model.predict(data.x_test)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(data.y_test.ravel(), label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()

plt.figure()
outputs = model.predict(data.x_train)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(data.y_train.ravel(), label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()