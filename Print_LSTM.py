import tensorflow as tf
import keras
from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
from keras.models import Model, Sequential
from keras.activations import relu, tanh
import numpy as np
import talos as ta
from talos.utils import live
from functions import *
from create_raw_data import *
from talos.utils import lr_normalizer, best_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from functions import mse
from keras.optimizers import SGD, Adam, Nadam
from keras.utils import plot_model

#res = pd.dataframe(
columns=['RMSE','MAPE','PCT','MAE','AMAPE']
#windows = [3, 5, 7, 10, 15, 20]
windows = [20,30]

res = np.zeros((len(windows),len(columns)))
ind = 0

for i in windows:
    df = create_finance(returns=False, plot_corr=False, Trends=True)
    data = prepare_data(df, i, 10, 10, returns=False, normalize_cheat=False)
    data.create_windows()
    x_train = data.x_train
    y_train = data.y_train.reshape(-1, 1)
    x_valid = data.x_valid
    y_valid = data.y_valid.reshape(-1, 1)
    x_test = data.x_test
    y_test = data.y_test.reshape(-1, 1)

#    file = 'ANN_WithoutTrends_stock_window_' + str(i)
    file = 'Talos_Results/LSTM_window_large_b' + str(i)

    print('<><><><><><><><><><><><><><><><><><><><><><><><><>')
    print("Opening Talos on window size: {}".format(i))
    dpl = file + '_deploy.zip'
    print(dpl)
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        LSTM = ta.Restore(dpl);
    #y_pred = ANN.model.predict(x_test)
    #f = '/Users/mariusjessen/UiO19/MachineLearning/HW/FYS-STK-4155-Project3/my_dir/'+file + '.png'
    #plot_model(ANN.model, to_file=f,show_shapes=True)


    y_pred = data.inv.inverse_transform(LSTM.model.predict(x_test))
    y_test = data.inv.inverse_transform(y_test)

    #y_test, y_pred = y_test[:20], y_pred[:20]




    print('RMSE: {}'.format(np.sqrt(mse(y_test,y_pred))))
    print('MAPE: {}'.format(mean_absolute_percentage_error(y_test,y_pred)))
    print('AMAPE: {}'.format(AMAPE(y_pred,y_test)))
    print('MAE: {}'.format(MAE(y_pred,y_test)))
    print('PCT: {}'.format(PCT(y_test,y_pred)))

    rmse=np.sqrt(mse(y_test,y_pred))
    mape=mean_absolute_percentage_error(y_test,y_pred)
    amape = AMAPE(y_pred,y_test)
    mae = MAE(y_pred,y_test)
    pct = PCT(y_test,y_pred)
    res[ind,0] = rmse
    res[ind,1] = mape
    res[ind,2] = pct
    res[ind,3] = mae
    res[ind,4] = amape
    ind = ind+1



res_df = pd.DataFrame(data=res,columns=columns)
print(res_df.to_latex(columns=columns))
   # print('RMSE: {}'.format(sqrt)
#10 with
#20 without

    #print(mse(y_valid, y_pred))

    #best_model = ANN.details.best_model(metric='mean_absolute_error', asc=True)