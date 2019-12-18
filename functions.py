from pandas import read_csv
import numpy as np
from create_raw_data import *
from sklearn.metrics import make_scorer
import pandas as pd
from pandas import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from numpy.random import seed
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return np.mean(np.abs((y_pred - y_true) / y_true))*100

def PCT(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    phi = np.zeros((len(y_true.ravel())-1))
    for i in range(len(y_true)-1):
        if(((y_pred[i+1]-y_true[i])*(y_true[i+1]-y_true[i]))>0):
            phi[i] = 1
        else:
            phi[i] = 0
    return np.mean(phi)


def mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return sum((y_true-y_pred)**2)/len(y_true)



class prepare_data():
        def __init__(self,df,window_length=20,test_percent=20, valid_percent=10, returns = False,normalize_cheat=False,Trends=True):
            self.test_percent = test_percent
            self.valid_percent = valid_percent
            self.data = df
            self.returns = returns
            self.window_length = window_length
            self.cheat = normalize_cheat
            if Trends:
                self.trends = 6
            else:
                self.trends = 5

        def normalize_windows(self):
            sc = MinMaxScaler()
            sc2 = StandardScaler(with_std= False)
            for i in range(len(self.x_train)):
                self.x_train[i] = sc.fit_transform(self.x_train[i])
                self.x_train[i] = sc2.fit_transform(self.x_train[i])

            for i in range(len(self.x_valid)):
                self.x_valid[i] = sc.transform(self.x_valid[i])
                self.x_valid[i] = sc2.transform(self.x_valid[i])

            for i in range(len(self.x_test)):
                self.x_test[i] = sc.transform(self.x_test[i])
                self.x_test[i] = sc2.transform(self.x_test[i])


            self.y_train = sc.fit_transform(self.y_train.reshape(-1, 1))
            self.y_valid = sc.transform(self.y_valid.reshape(-1, 1))
            self.y_test = sc.transform(self.y_test.reshape(-1, 1))

            self.mm = [sc.data_min_, sc.data_max_]

        #Important to normalize train and test data for themselves
        def normalize(self):

            sc = MinMaxScaler()
            self.x_train = sc.fit_transform(self.x_train)
            self.x_valid = sc.transform(self.x_valid)
            self.x_test = sc.transform(self.x_test)



            sc2 = StandardScaler(with_std=False)
            self.x_train = sc2.fit_transform(self.x_train)
            self.x_valid = sc2.transform(self.x_valid)
            self.x_test = sc2.transform(self.x_test)


#            print(self.y_train.shape)

            if self.returns == True:
                self.y_train =self.y_train.reshape(-1,1)
                self.y_valid = self.y_valid.reshape(-1,1)
                self.y_test = self.y_test.reshape(-1,1)
            else:
                self.y_train = sc.fit_transform(self.y_train.reshape(-1,1))
                self.y_valid = sc.transform(self.y_valid.reshape(-1,1))
                self.y_test = sc.transform(self.y_test.reshape(-1,1))
                self.mm = [sc.data_min_, sc.data_max_]



        def normalize_all(self):
            sc = MinMaxScaler()
            sc2 = StandardScaler(with_std=False)
            sc3 = StandardScaler()
            self.data = sc.fit_transform(self.data)
            #self.data = sc2.fit_transform(self.data[:-1])
            self.mm = [sc.data_min_[-1], sc.data_max_[-1]]
            #self.mm = [sc.data_min_, sc.data_max_]
            #self.data = sc2.fit_transform(self.data)




        def create_windows(self):

            pca = PCA(n_components=0.95)
            window_length = self.window_length
            data_matrix = np.empty(1)
            if self.cheat == True:
                self.normalize_all()
                data_matrix = self.data
            if self.cheat != True:
                data_matrix = self.data.as_matrix()
            stock_windows_train = []
            stock_windows_valid = []
            stock_windows_test = []
            valid_size = int(np.round(self.valid_percent / 100 * len(data_matrix)))
            test_size = int(np.round(self.test_percent / 100 * len(data_matrix)))
            train_size = len(data_matrix) - (test_size + valid_size)



            self.x_train = data_matrix[:train_size, :-1]
            self.y_train = data_matrix[window_length-1:train_size,  -1]

            self.x_valid = data_matrix[train_size:train_size + valid_size, :-1]
            self.y_valid = data_matrix[window_length-1+train_size:train_size + valid_size, -1]

            self.x_test = data_matrix[train_size + valid_size:,:-1]
            self.y_test = data_matrix[window_length-1+train_size + valid_size:, -1]
            if self.cheat == False:
                self.normalize()

            x_train_pca = pca.fit_transform(self.x_train[:,self.trends:])
            x_valid_pca = pca.transform(self.x_valid[:,self.trends:])
            x_test_pca = pca.transform(self.x_test[:,self.trends:])

            self.x_train = np.c_[self.x_train[:,:self.trends],x_train_pca]
            self.x_valid = np.c_[self.x_valid[:, :self.trends], x_valid_pca]
            self.x_test = np.c_[self.x_test[:, :self.trends], x_test_pca]


            #Create all possibl e sequences
            for index in range(len(x_train_pca)-window_length+1):
                stock_windows_train.append(self.x_train[index:index+window_length])

            self.x_train = np.array(stock_windows_train) #(X_inputs,window_size,indicators)


            for index in range(len(x_valid_pca)-window_length+1):
                stock_windows_valid.append(self.x_valid[index:index+window_length])

            self.x_valid = np.array(stock_windows_valid) #(X_inputs,window_size,indicators)


            for index in range(len(x_test_pca)-window_length+1):
                stock_windows_test.append(self.x_test[index:index+window_length])

            self.x_test = np.array(stock_windows_test) #(X_inputs,window_size,indicators)

            #self.normalize_windows()
            #Close price adjusted in raw data, thus up to last value



        def normalized_windows(self):

            pca = PCA(n_components=0.95)
            window_length = self.window_length
            data_matrix = np.empty(1)

            data_matrix = self.data.as_matrix()
            stock_windows_train = []
            stock_windows_valid = []
            stock_windows_test = []

            valid_size = int(np.round(self.valid_percent / 100 * len(data_matrix)))
            test_size = int(np.round(self.test_percent / 100 * len(data_matrix)))
            train_size = len(data_matrix) - (test_size + valid_size)

            #Scale train first and create windows

            self.x_train = data_matrix[:train_size, :-1]
            self.y_train = data_matrix[window_length - 1:train_size, -1]

            sc = StandardScaler()
            self.x_train = sc.fit_transform(self.x_train)
            self.y_train = sc.fit_transform(self.y_train.reshape(-1,1))
            self.train_sc = sc

            x_train_pca = pca.fit_transform(self.x_train[:, self.trends:])
            self.x_train = np.c_[self.x_train[:, :self.trends], x_train_pca]

            for index in range(len(x_train_pca)-window_length+1):
                stock_windows_train.append(self.x_train[index:index+window_length])

            self.x_train = np.array(stock_windows_train)

            #Now scale test values, but now scale each window instead
            self.x_valid = data_matrix[train_size:train_size + valid_size, :-1]
            self.y_valid = data_matrix[window_length - 1 + train_size:train_size + valid_size, -1]

            x_valid_scaled, mu_val_list, std_val_list = [], [], []
            #x_val = self.x_valid
            for index in range(len(self.x_valid) - window_length + 1):
                #mu_val_list.append(np.mean(x_val[index:index + window_length],axis=1))
                #std_val_list.append(np.std(x_val[index:index+ window_length],axis=1))
                sc = StandardScaler()
                x_val = sc.fit_transform(self.x_valid[index:index + window_length])

                x_val_pca = pca.transform(x_val[:, self.trends:])
                x_val = np.c_[x_val[:, :self.trends], x_val_pca]

                #x_val_pca = pca.fit_transform(x_val[index:index + window_length, self.trends:])

                #x_val = np.c_[x_val[index:index + window_length, :self.trends], x_val_pca]

                x_valid_scaled.append(x_val)

            x_valid_scaled = np.array(x_valid_scaled)

            self.x_valid = x_valid_scaled

            sc = StandardScaler()
            self.y_valid = sc.fit_transform(self.y_valid.reshape(-1,1))
            self.valid_sc = sc
            #self.mu_val_list = mu_val_list
            #self.std_val_list = std_val_list

            self.x_test = data_matrix[train_size + valid_size:, :-1]
            self.y_test = data_matrix[window_length - 1 + train_size + valid_size:, -1]

            x_test_scaled, mu_test_list, std_test_list = [], [], []
            #x_test = self.x_test
            for index in range(len(self.x_test) - window_length + 1):
                #mu_test_list.append(np.mean(x_val[index:index + window_length]))
                #std_test_list.append(np.std(data[index:+ window_length]))
                sc = StandardScaler()
                x_test = sc.fit_transform(self.x_test[index:index + window_length])

                #x_test = (x_test[index:index + window_length] - mu_test_list[index]) / std_test_list[index]

                x_test_pca = pca.transform(x_test[:, self.trends:])
                x_test = np.c_[x_test[:, :self.trends], x_test_pca]

                x_test_scaled.append(x_test)

            self.x_test = np.array(x_test_scaled)
            sc = StandardScaler()
            self.y_test = sc.fit_transform(self.y_test.reshape(-1,1))
            self.test_sc = sc

            #self.mu_test_list = mu_test_list
            #self.std_test_list = std_test_list



def create_ANN(data, lr_=0.001, dropout_rate = 0.2):
    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=128, return_sequences=True, input_shape=(data.x_train.shape[1], data.x_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=50))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    opt = Adam(lr=lr_)
    model.compile(optimizer=opt, loss='mean_squared_error',metrics=['mean_squared_error'])
    return model



def run_Grid(data):

    model= KerasRegressor(build_fn=create_ANN(data),verbose=0)
    params = {'lr_': [0.001, 0.01, 0.1, 0.2, 0.3],
              'epochs': [100, 150, 200],
              'batch_size': [100, 150],
              'dropout_rate': [0.3],
            'validation_data': [(data.x_valid, data.y_valid.ravel())]}


    regressor = GridSearchCV(estimator = model, param_grid = params, n_jobs = 1, refit=True, scoring='neg_mean_squared_error')
    regressor.fit(data.x_train, data.y_train.ravel())
    """
    preds = regressor.predict(data.x_test)
    print("Best: %f using %s" % (regressor.best_score_, regressor.best_params_))
    """





if __name__ == "__main__":
    print('Running stock predictor test')

    df = create_finance(returns=False, plot_corr=False, Trends=True)

    # Create sliding windows
    data = prepare_data(df, 30, 20, 10, returns=False, normalize_cheat=False)

    # Sliding windows of 20 days
    data.normalized_windows()

    print(data.x_test.shape)


