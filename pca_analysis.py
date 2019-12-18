import tensorflow as tf
import keras
from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
from keras.models import Model, Sequential
from keras.activations import relu, tanh
import talos as ta
from talos.utils import live
from functions import *
from create_raw_data import *
from talos.utils import lr_normalizer, best_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from functions import mse
from keras.optimizers import SGD, Adam, Nadam

df = create_finance(returns=False, plot_corr=False, Trends=True)
print(df.columns)

indicators = ['ACD', 'VPT', 'MACD', 'CHO', 'STO_k', 'STO_d', 'PROC', 'VROC', 'OBV', 'WR', 'RSI', 'MOM', 'AC']
data_df = df[indicators]

data = data_df.as_matrix()
valid_size = int(np.round(10 / 100 * len(data)))
test_size = int(np.round(10 / 100 * len(data)))
train_size = len(data) - (test_size + valid_size)

x_train = data[:train_size, :]
x_valid = data[train_size:train_size + valid_size, :]
x_test = data[train_size + valid_size:, :]


sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)

sc2 = StandardScaler(with_std=False)
x_train = sc2.fit_transform(x_train)
x_valid = sc2.transform(x_valid)
x_test = sc2.transform(x_test)


pca = PCA(n_components=0.95)
x_train_pca = pca.fit_transform(x_train)
x_valid_pca = pca.transform(x_valid)
x_test_pca = pca.transform(x_test)
print(x_train.shape)
print(x_train_pca.shape)

print(df[indicators].iloc[:1])
print(x_valid[0,:])
print(x_valid_pca[0,:])