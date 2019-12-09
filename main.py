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
from sklearn.model_selection import learning_curve
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

"""------------------------- Data processing -----------------------------"""


#Create dataset
df = create_finance(returns=True)

#Create sliding windows
data = prepare_data(df,20,20,10)

#Sliding windows of 20 days
data.create_windows(20)

"""------------------------- Functions to be runned -----------------------------"""

def run_SVR(data):
    x_train = data.x_train.reshape(len(data.x_train),-1)
    x_valid = data.x_valid.reshape(len(data.x_valid),-1)
    x_test = data.x_test.reshape(len(data.x_test),-1)

    svr = SVR(kernel='poly',C=10,verbose=True)

    svr.fit(x_train,data.y_train.ravel())

    predictions = svr.predict(x_test)

    print(svr.score(x_test,data.y_test.ravel()))

    plt.figure()
    plt.plot(data.y_test.ravel(), label='targets')
    plt.plot(predictions, label='predictions')
    plt.legend()
    plt.title('SVR test data')
    plt.show()

    predictions = svr.predict(x_train)

    plt.plot(data.y_train.ravel(), label='targets')
    plt.plot(predictions, label='predictions')
    plt.legend()
    plt.title('SVR train data')
    plt.show()


    #Create plots
    plt.figure()
    plt.plot(svr.loss_curve_, label='val_loss')
    plt.title('SVR loss')
    plt.legend()


def run_LSTM(data):
    #build model
    i = Input(shape=(20, 14))
    x = LSTM(20)(i)
    x = Dense(1)(x)
    model = Model(i, x)
    model.compile(
      loss='mse',
      optimizer=Adam(lr=0.01),
    )
    # train the RNN
    r = model.fit(
      data.x_train, data.y_train.ravel(),
      epochs=80,
      validation_data=(data.x_valid,data.y_valid.ravel()),
    )
    #Create plots
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.title('LSTM')
    plt.legend()

    plt.figure()
    outputs = model.predict(data.x_test)
    print(outputs.shape)
    predictions = outputs[:,0]

    plt.plot(data.y_test.ravel(), label='targets')
    plt.plot(predictions, label='predictions')
    plt.legend()
    plt.title('LSTM test data')
    plt.show()

    plt.figure()
    outputs = model.predict(data.x_train)
    print(outputs.shape)
    predictions = outputs[:,0]

    plt.plot(data.y_train.ravel(), label='targets')
    plt.plot(predictions, label='predictions')
    plt.legend()
    plt.title('LSTM train data')
    plt.show()


def run_ANN(data):
    x_train = data.x_train.reshape(len(data.x_train),-1)
    x_valid = data.x_valid.reshape(len(data.x_valid),-1)
    x_test = data.x_test.reshape(len(data.x_test),-1)
    # Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(280,), activation='relu'),
        tf.keras.layers.Dense(1)])

    # Compile and fit
    opt = tf.keras.optimizers.Adam(0.01)
    model.compile(optimizer=opt, loss='mse')
    r = model.fit(x_train, data.y_train.ravel(), epochs=100, validation_data=(x_valid,data.y_valid.ravel()))

    #Create plots
    plt.figure()
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.title('ANN loss')
    plt.legend()


    outputs = model.predict(x_test)
    print(outputs.shape)
    predictions = outputs[:,0]

    plt.figure()
    plt.plot(data.y_test.ravel(), label='targets')
    plt.plot(predictions, label='predictions')
    plt.legend()
    plt.title('ANN test data')
    plt.show()

    plt.figure()
    outputs = model.predict(x_train)
    print(outputs.shape)
    predictions = outputs[:,0]

    plt.plot(data.y_train.ravel(), label='targets')
    plt.plot(predictions, label='predictions')
    plt.legend()
    plt.title('ANN train data')
    plt.show()


"""------------------------- Run Code -----------------------------"""
#run_SVR(data)
#run_ANN(data)
run_LSTM(data)