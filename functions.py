from pandas import read_csv
import numpy as np
import pandas as pd
from pandas import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# **** change the warning level ****



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return np.mean(np.abs((y_pred - y_true) / y_true))*100

def PCT(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    phi = np.zeros((y_true.ravel().shape))
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
                self.trends = 7
            else:
                self.trends = 6

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

            print(self.y_train.shape)

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
            print(data_matrix.shape)
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

            class BlockingTimeSeriesSplit():
                def __init__(self, n_splits):
                    self.n_splits = n_splits

                def get_n_splits(self, X, y, groups):
                    return self.n_splits

                def split(self, X, y=None, groups=None):
                    n_samples = len(X)
                    k_fold_size = n_samples // self.n_splits
                    indices = np.arange(n_samples)

                    margin = 0
                    for i in range(self.n_splits):
                        start = i * k_fold_size
                        stop = start + k_fold_size
                        mid = int(0.8 * (stop - start)) + start
                        yield indices[start: mid], indices[mid + margin: stop]







if __name__ == "__main__":
    print('Running stock predictor test')