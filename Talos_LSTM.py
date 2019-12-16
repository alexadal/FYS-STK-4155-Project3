"""
Optimizer for tensoflow.Keras --> used for grid search
"""
import keras
import tensorflow as tf
#from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
#from keras.models import Model, Sequential
#from keras.activations import relu, tanh
from sklearn.metrics import mean_squared_error
from math import sqrt
import talos as ta
from talos.utils import live
from functions import *
from create_raw_data import *
from talos.utils import lr_normalizer
#from keras.optimizers import SGD, Adam, Nadam
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

seed(1)
#random.set_seed(2)


#128,256
# set the parameter space
p = {'lr': (1e-3, 1e-1, 15),
     'first_neuron':[10,50,128,256],
    'second_neuron':[10,50,128,256],
     'batch_size': [10,20,50,100],
     'epochs': [20, 30, 50, 90],
     'dropout': [0, 0.2, 0.4],
     'optimizer': [Adam],
     'losses': ['mse']}

# first we have to make sure to input data and params into the function
def create_model(trainX, trainY, testX, testY, params):

    model = Sequential()
    model.add(LSTM(units=params['first_neuron'], return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(units=params['first_neuron']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(units=params['second_neuron']))
    model.add(Dense(units=1))

    model.compile(optimizer=params['optimizer'](lr=params['lr']),loss=['mean_squared_error'], metrics=['mean_squared_error'])

    model_out = model.fit(trainX, trainY,
                        validation_data=[testX, testY],
                        batch_size=params['batch_size'],
                        callbacks=[tf.keras.callbacks.History()],
                        epochs=params['epochs'],
                        verbose=0)



    return model_out, model

def get_best_Talos(windows):

    df = create_finance(returns=False, plot_corr=False, Trends=True)

    for i in windows:
        data = prepare_data(df, i, 10, 10, returns=False, normalize_cheat=False)
        data.create_windows()

        x_train = data.x_train
        y_train = data.y_train.reshape(-1, 1)

        x_valid = data.x_valid
        y_valid = data.y_valid.reshape(-1, 1)

        file = 'LSTM_window' + str(i)
        print("Running Talos on window size: {}".format(i))

        t = ta.Scan(x=x_train,y=y_train,x_val=x_valid,y_val=y_valid, model=create_model, params=p,experiment_name=file,fraction_limit=0.01)
        dpl = file+'_deploy'
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            ta.Deploy(scan_object=t, model_name=dpl, metric='val_mean_squared_error',asc=True);







if __name__ == "__main__":

    df = create_finance(returns=False, plot_corr=False, Trends=True)

    # Create sliding windows
    data = prepare_data(df, 3, 10, 10, returns=False, normalize_cheat=False)

    # Sliding windows of 20 days
    data.create_windows()

    x_train = data.x_train
    y_train = data.y_train.reshape(-1,1)

    x_valid = data.x_valid
    y_valid = data.y_valid.reshape(-1,1)

    x_test = data.x_test
    y_test = data.y_test.reshape(-1,1)
    

    windows = [3,5,7,10,15,20]
    #get_best_Talos(windows)

    """
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):

        LSTM = ta.Restore('LSTM_stock_model_window3_deploy.zip');

    y_pred = LSTM.model.predict(x_test)

    print(y_pred)

    y_true, y_pred = np.array(data.y_test).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)

    plt.figure()
    plt.plot(data.y_test.ravel(), label='targets')
    plt.plot(y_pred, label='predictions')
    plt.legend()
    plt.title('LSTM test data')
    plt.show()

    mape = mean_absolute_percentage_error(y_true, y_pred)
    pct = PCT(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)

    print('MAPE = {:.2f}, PCT = {:.2f}, MSE = {:.6f} and RMSE = {:.5f}'.format(mape, pct,mse,rmse))

    """

