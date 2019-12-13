import tensorflow as tf
import keras
from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
from keras.models import Model, Sequential
from keras.activations import relu, tanh
import talos as ta
from talos.utils import live
from functions import *
from create_raw_data import *
from talos.utils import lr_normalizer
from keras.optimizers import SGD, Adam, Nadam
from talos.utils.recover_best_model import recover_best_model

df = create_finance(returns=False, plot_corr=False, Trends=True)

# Create sliding windows
data = prepare_data(df, 30, 20, 10, returns=False, normalize_cheat=False)

# Sliding windows of 20 days
data.create_windows()

x_train = data.x_train.reshape(len(data.x_train), -1)
y_train = data.y_train.reshape(-1, 1)

x_valid = data.x_valid.reshape(len(data.x_valid), -1)
y_valid = data.y_valid.reshape(-1, 1)

x_test = data.x_test.reshape(len(data.x_test), -1)
y_test = data.y_test.reshape(-1, 1)

results, models = recover_best_model(x_train=x_train,
                                     y_train=y_train,
                                     x_val=x_val,
                                     y_val=y_val,
                                     experiment_log='/ANN Stock/121319230413.csv',
                                     input_model=iris_model,
                                     n_models=5,
                                     task='multi_label')


