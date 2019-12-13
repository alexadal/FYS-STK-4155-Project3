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
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from numpy.random import seed
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor


seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

"""------------------------- Data processing -----------------------------"""

Returns = False
#Create dataset
df = create_finance(returns=Returns,plot_corr=False,Trends=False)

#Create sliding windows
data = prepare_data(df,45,20,10,returns=False,normalize_cheat=False)

#Sliding windows of 20 days
#data.create_windows()
data.normalized_windows()

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
    """
    #build model
    i = Input(shape=(data.x_train.shape[1], data.x_train.shape[2]))
    x = LSTM(50)(i)
    x = Dense(1)(x)
    model = Model(inputs=i, outputs=x)
    model.compile(
      loss='mse',
      optimizer=Adam(lr=0.1,decay=0.2),
    )
    # train the RNN
    regressor = model.fit(
      data.x_train, data.y_train.ravel(),
      epochs=50, verbose=True, shuffle=True
      ,validation_data=(data.x_valid,data.y_valid.ravel()))
    """

    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=20, input_shape=(data.x_train.shape[1], data.x_train.shape[2])))
    regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units=1))
    # Compiling the RNN
    opt = Adam(lr=0.3)
    regressor.compile(optimizer=opt, loss='mean_squared_error',metrics=['mean_absolute_percentage_error'])

    # Fitting the RNN to the Training set
    regressor.fit(data.x_train, data.y_train.ravel(), epochs=80,validation_data=(data.x_valid,data.y_valid.ravel()))

    #Create plots
    plt.plot(regressor.history['loss'], label='loss')
    plt.plot(regressor.history['val_loss'], label='val_loss')
    plt.plot(regressor.history['mean_absolute_percentage_error'], label='MAPE')
    plt.title('LSTM')
    plt.legend()
    plt.show()

    """"
    plt.figure()
    outputs = model.predict(data.x_test)
    print(outputs.shape)
    predictions = outputs[:,0]

    print(predictions.shape)

    pred_prices = predictions.reshape(-1)
    real_prices = data.y_test.reshape(-1)
    mape = 0


    if Returns == False:
        
        mape = mean_absolute_percentage_error(real_prices, pred_prices)

    #pred_prices = [x * (data.mm[1] - data.mm[0]) + data.mm[0] for x in predictions.reshape(-1)]
    #real_prices = [x * (data.mm[1] - data.mm[0]) + data.mm[0] for x in data.y_test.reshape(-1)]

    mape = mean_absolute_percentage_error(data.y_test.ravel(), predictions.ravel())
    y_true, y_pred = np.array(real_prices), np.array(pred_prices)
    #mape = mean_absolute_percentage_error(y_true.reshape(-1,1), y_pred.reshape(-1,1))
    pct = PCT(real_prices,pred_prices)
    mse = mean_squared_error(real_prices,pred_prices)
    rmse = sqrt(mse)

    plt.plot(real_prices, label='targets')
    plt.plot(pred_prices, label='predictions')
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

    print('MAPE = {:.2f}, PCT = {:.2f}, MSE = {:.6f} and RMSE = {:.5f}'.format(mape, pct,mse,rmse))
    """


def run_ANN(data):
    x_train = data.x_train.reshape(len(data.x_train),-1)
    x_valid = data.x_valid.reshape(len(data.x_valid),-1)
    x_test = data.x_test.reshape(len(data.x_test),-1)
    # Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(50, input_shape=(len(x_train[1]),), activation='relu'),
        tf.keras.layers.Dense(1)])

    # Compile and fit
    opt = tf.keras.optimizers.Adam(0.05)
    model.compile(optimizer=opt, loss='mse')
    r = model.fit(x_train, data.y_train.ravel(), epochs=100, validation_data=(x_valid,data.y_valid.ravel()))

    #Create plots
    plt.figure()
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.title('ANN loss')
    plt.legend()
    plt.show()


    outputs = model.predict(x_test)
    print(outputs.shape)
    predictions = outputs[:,0]

    pred_prices = predictions.reshape(-1)
    real_prices = data.y_test.reshape(-1)

    if Returns != True:
        pred_prices = [x*(data.mm[0]-data.mm[1])+data.mm[0] for x in predictions.reshape(-1)]
        real_prices = [x * (data.mm[0] - data.mm[1]) + data.mm[0] for x in data.y_test.reshape(-1)]


    mape = mean_absolute_percentage_error(data.y_test.ravel(),predictions.ravel())
    pct = PCT(data.y_test.ravel(),predictions.ravel())



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


    print('MAPE = {:.2f} and PCT = {:.2f}'.format(mape, pct))


"""------------------------- Run Code -----------------------------"""
#run_SVR(data)
#run_ANN(data)
run_LSTM(data)