from pandas import read_csv
import numpy as np
from pandas import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# **** change the warning level ****


def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass




class prepare_data():
        def __init__(self,df,window_lenght,test_percent=20, valid_percent=10, Normalize = 'MM'):
            self.test_percent = test_percent
            self.valid_percent = valid_percent
            self.data = df
            self.Normalize = Normalize

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

        #Important to nomralize train and test data for themselves
        def normalize(self):
            if self.Normalize == 'MM':
                sc = MinMaxScaler()
                self.x_train = sc.fit_transform(self.x_train)
                self.x_valid = sc.transform(self.x_valid)
                self.x_test = sc.transform(self.x_test)

                sc2 = StandardScaler(with_std=False)
                self.x_train = sc2.fit_transform(self.x_train)
                self.x_valid = sc2.transform(self.x_valid)
                self.x_test = sc2.transform(self.x_test)

                print(self.y_train.shape)
                self.y_train = sc.fit_transform(self.y_train.reshape(-1,1))
                self.y_valid = sc.transform(self.y_valid.reshape(-1,1))
                self.y_test = sc.transform(self.y_test.reshape(-1,1))

                self.mm = [sc.data_min_, sc.data_max_]

        def create_windows(self,window_length):

            pca = PCA(n_components=0.95)


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

            self.normalize()

            x_train_pca = pca.fit_transform(self.x_train[:,7:])
            x_valid_pca = pca.transform(self.x_valid[:,7:])
            x_test_pca = pca.transform(self.x_test[:,7:])

            self.x_train = np.c_[self.x_train[:,:7],x_train_pca]
            self.x_valid = np.c_[self.x_valid[:, :7], x_valid_pca]
            self.x_test = np.c_[self.x_test[:, :7], x_test_pca]



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
            #Assume close price adjusted in raw data, thus up to last value


































if __name__ == "__main__":
    print('Running stock predictor test')