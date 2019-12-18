"""
Optimizer for tensoflow.Keras --> used for grid search
"""
import keras
import tensorflow as tf
#from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
#from keras.models import Model, Sequential
#from keras.activations import relu, tanh
import talos as ta
from talos.utils import live
from functions import *
from create_raw_data import *
from talos.utils import lr_normalizer
#from keras.optimizers import SGD, Adam, Nadam
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform



# set the parameter space
p = {'lr': (1e-5, 1, 20),
     'first_neuron':[20, 50, 128],
     'second_neuron':[20, 50, 128],
     'batch_size': [10, 20, 50, 100, 300],
     'epochs': [10, 30, 50, 100],
     'dropout': [0, 0.2],
     'optimizer': [Adam],
     'losses': ['mse']}

# first we have to make sure to input data and params into the function
def create_model(trainX, trainY, testX, testY, params):
    model = Sequential([
        Dense(params['first_neuron'], input_shape=(len(trainX[1]),), activation='relu'),
        Dense(1)])

    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=['mean_squared_error'], metrics=['mean_squared_error'])

    model_out = model.fit(trainX, trainY,
                        validation_data=[testX, testY],
                        batch_size=params['batch_size'],
                        callbacks=[keras.callbacks.History()],
                        epochs=params['epochs'],
                        verbose=0)



    return model_out, model

def get_best_Talos(windows):

    df = create_finance(returns=False, plot_corr=False, Trends=False)

    for i in windows:
        data = prepare_data(df, i, 10, 10, returns=False, normalize_cheat=False)
        data.create_windows()

        x_train = data.x_train.reshape(len(data.x_train),-1)
        y_train = data.y_train.reshape(-1, 1)

        x_valid = data.x_valid.reshape(len(data.x_valid),-1)
        y_valid = data.y_valid.reshape(-1, 1)

        file = 'Talos_Results/ANN_WithoutTrends_stock_window_' + str(i)
        print("Running Talos on window size: {}".format(i))

        t = ta.Scan(x=x_train,y=y_train,x_val=x_valid,y_val=y_valid, model=create_model, params=p,experiment_name=file,fraction_limit=0.1,seed=2)
        dpl = file+'_deploy'
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            ta.Deploy(scan_object=t, model_name=dpl, metric='mean_squared_error', asc=True);


if __name__ == "__main__":
    windows = [1, 3, 5, 7, 10, 15, 20]
    #windows = [1, 20]
    get_best_Talos(windows)
