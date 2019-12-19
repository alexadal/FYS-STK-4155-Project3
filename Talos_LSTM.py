"""
Optimizer for tensoflow.Keras --> used for grid search
"""
import keras
import tensorflow as tf
#from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
#from keras.models import Model, Sequential
#from keras.activations import relu, tanh
from sklearn.metrics import mean_squared_error
from IPython.display import SVG
from keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot

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
p = {'lr': (1e-4, 1e-2, 20),
     #'first_neuron':[10,50,128,256],
    #'second_neuron':[128,256],
     'batch_size': [300,500,600,800],
     'epochs': [300, 400, 500],
     'optimizer': [Adam],
     'losses': ['mse']}

# first we have to make sure to input data and params into the function
def create_model(trainX, trainY, testX, testY, params):
    """
    model = Sequential()
    model.add(LSTM(units=params['first_neuron'], return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(units=params['first_neuron']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(units=params['second_neuron']))
    model.add(Dense(units=1))
    """
    model = Sequential()
    model.add(LSTM(units=10, return_sequences=False, input_shape=(trainX.shape[1], trainX.shape[2])))
    #model.add(Dropout(params['dropout']))
    #model.add(LSTM(units=10))
    #model.add(Dropout(params['dropout']))
    #model.add(Dense(units=20))
    model.add(Dense(units=1,activation='relu',kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=None)))
    # Compiling the RNN
    #opt = Adam(lr=0.00231)
    #model.compile(optimizer=opt, loss='mean_squared_error',metrics=['mean_absolute_percentage_error'])

    model.compile(optimizer=params['optimizer'](lr=params['lr']),loss=['mean_squared_error'], metrics=['mean_squared_error'])

    model_out = model.fit(trainX, trainY,
                        validation_data=[testX, testY],
                        batch_size=params['batch_size'],
                        callbacks=[tf.keras.callbacks.History()],
                        epochs=params['epochs'],shuffle=True,
                        verbose=0)



    return model_out, model

def get_best_Talos(windows):

    df = create_finance(returns=False, plot_corr=False, Trends=False)

    for i in windows:
        data = prepare_data(df, i, 10, 10, returns=False, normalize_cheat=False)
        data.create_windows()

        x_train = data.x_train
        y_train = data.y_train.reshape(-1, 1)

        x_valid = data.x_valid
        y_valid = data.y_valid.reshape(-1, 1)

        file = 'LSTM_window_large_repl_W_O' + str(i)
        print("Running Talos on window size: {}".format(i))

        t = ta.Scan(x=x_train,y=y_train,x_val=x_valid,y_val=y_valid, model=create_model, params=p,experiment_name=file,fraction_limit=0.1,seed=4)
        dpl = file+'_deploy'
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            ta.Deploy(scan_object=t, model_name=dpl, metric='val_mean_squared_error',asc=True);







if __name__ == "__main__":

    df = create_finance(returns=False, plot_corr=False, Trends=True)
    df2 = create_finance(returns=False, plot_corr=False, Trends=False)

    # Create sliding windows
    data = prepare_data(df, 20, 10, 10, returns=False, normalize_cheat=False)
    data2 = prepare_data(df2, 20, 10, 10, returns=False, normalize_cheat=False)

    # Sliding windows of 20 days
    data.create_windows()
    data2.create_windows()



    x_train = data.x_train
    y_train = data.y_train.reshape(-1,1)

    x_valid = data.x_valid
    y_valid = data.y_valid.reshape(-1,1)


    x_test2 = data2.x_test
    x_test = data.x_test
    y_test = data2.y_test.reshape(-1,1)
    

    windows = [15,20,30]
    #get_best_Talos(windows)


    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):

        #LSTM = ta.Restore('LSTM_window20_deploy.zip');
        #LSTM2 = ta.Restore('LSTM_window_W_O_Trend20_deploy.zip');

        #LSTM = ta.Restore('LSTM_window_W_O_Trend15_deploy.zip');
        #LSTM = ta.Restore('LSTM_window_large_b220_deploy.zip');
        #LSTM2 = ta.Restore('LSTM_window_large_b2_WO_Trend220_deploy.zip');
        LSTM = ta.Restore('LSTM_window_large_repl20_deploy.zip');
        LSTM2 = ta.Restore('LSTM_window_large_repl_W_O20_deploy.zip');




    y_pred = LSTM.model.predict(x_test)
    y_pred2 = LSTM2.model.predict(x_test2)

    #plot_model(LSTM.model, to_file='LSTM_20_model.png')
    #SVG(model_to_dot(LSTM.model).create(prog='dot', format='svg'))



    print(y_pred)

    y_true, y_pred = np.array(data.y_test).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)
    y_pred2 = np.array(y_pred2).reshape(-1, 1)


    #y_pred = [x * (data.mm[1] - data.mm[0]) + data.mm[0] for x in y_pred.reshape(-1)]
    #y_true = [x * (data.mm[1] - data.mm[0]) + data.mm[0] for x in y_true.reshape(-1)]

    y_pred = data.inv.inverse_transform(y_pred)
    y_pred2 = data.inv.inverse_transform(y_pred2)

    y_true = data.inv.inverse_transform(y_true)

    plt.figure()
    plt.plot(y_true, label='targets')
    plt.plot(y_pred, label='predictions')
    plt.plot(y_pred2, label='W_O trends')

    plt.legend()
    plt.title('LSTM test data')
    plt.show()

    mape = mean_absolute_percentage_error(y_true, y_pred)
    pct = PCT(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)

    mape2 = mean_absolute_percentage_error(y_true, y_pred2)
    pct2 = PCT(y_true, y_pred2)
    mse2 = mean_squared_error(y_true, y_pred2)
    rmse2 = sqrt(mse2)

    print('MAPE = {:.2f}, PCT = {:.2f}, MSE = {:.6f} and RMSE = {:.5f}'.format(mape, pct,mse,rmse))
    print('MAPE = {:.2f}, PCT = {:.2f}, MSE = {:.6f} and RMSE = {:.5f}'.format(mape2, pct2,mse2,rmse2))
