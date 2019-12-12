"""
Optimizer for tensoflow.Keras --> used for grid search
"""

from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from keras.activations import relu, tanh
import talos as ta
from talos.utils import live
from functions import *
from create_raw_data import *
from talos.utils import lr_normalizer
from tensorflow.keras.optimizers import SGD, Adam, Nadam
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# set the parameter space
p = {'lr': (1e-5, 0.1, 10),
     'first_neuron':[50, 128],
     'second_neuron':[50, 128],
     'batch_size': [50, 100],
     'epochs': [20, 30, 50],
     'dropout': [0, 0.2, 0.7],
     'optimizer': [Adam, Nadam],
     'losses': ['mse']}

# first we have to make sure to input data and params into the function
def create_model(trainX, trainY, testX, testY, params):

    model = Sequential()
    model.add(LSTM(units=params['first_neuron'], return_sequences=True, input_shape=(data.x_train.shape[1], data.x_train.shape[2])))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(units=params['second_neuron']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(units=1))

    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),loss=['mean_squared_error'], metrics=['mean_squared_error'])

    model_out = model.fit(trainX, trainY,
                        validation_data=[testX, testY],
                        batch_size=params['batch_size'],
                        callbacks=[tf.keras.callbacks.History()],
                        epochs=params['epochs'],
                        verbose=0)



    return model_out, model



if __name__ == "__main__":

    df = create_finance(returns=False, plot_corr=False, Trends=True)

    # Create sliding windows
    data = prepare_data(df, 30, 20, 10, returns=False, normalize_cheat=False)

    # Sliding windows of 20 days
    data.create_windows()

    x_train = data.x_train
    y_train = data.y_train.reshape(-1,1)

    x_valid = data.x_valid
    y_valid = data.y_valid.reshape(-1,1)

    x_test = data.x_test
    y_test = data.y_test.reshape(-1,1)


    scan_object = ta.Scan(x=x_train,y=y_train,x_val=x_valid,y_val=y_valid, model=create_model, params=p,experiment_name='LSTM Stock',fraction_limit=0.01)

