from pandas import read_csv
import numpy as np
from pandas import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()
with lstm_graph.as_default():


    class prepare_data():
        def __init__(self,df,window_lenght,test_percent=20, valid_percent=10, Normalize = 'MM'):
            self.test_percent = test_percent
            self.valid_percent = valid_percent
            self.data = df
            self.normalize = Normalize


        #Important to nomralize train and test data for themselves
        def normalize(self,x_train,x_valid,x_test,y_train,y_valid,y_test):
            if self.normalize == 'MM':
                sc = MinMaxScaler()
                x_train = sc.fit_transform(x_train)
                x_valid = sc.transform(x_valid)
                x_test = sc.transform(x_test)

                sc2 = StandardScaler(with_std=False)
                x_train = sc2.fit_transform(x_train)
                x_valid = sc2.transform(x_valid)
                x_test = sc2.transform(x_test)

                self.mm = [np.amin(y_train),np.amax(y_train)]

                y_train = sc.fit_transform(y_train)
                y_valid = sc.transform(y_valid)
                y_test = sc.transform(y_test)




        def create_windows(self,window_length):

            pca = PCA()

            data_matrix = self.data.asmatrix()
            stock_windows_train = []
            stock_windows_valid = []
            stock_windows_test = []

            valid_size = int(np.round(self.valid_percent / 100 * len(data_matrix[0])))
            test_size = int(np.round(self.test_percent / 100 * len(data_matrix[0])))
            train_size = len(data_matrix[0]) - (test_size + valid_size)

            x_train = data_matrix[:train_size, :-1]
            y_train = data_matrix[window_length:train_size,  -1]

            x_valid = data_matrix[train_size:train_size + valid_size, :-1]
            y_valid = data_matrix[window_length+train_size:train_size + valid_size, -1]

            x_test = data_matrix[train_size + valid_size:,:-1]
            y_test = data_matrix[window_length+train_size + valid_size:, -1]

            x_train_pca = pca.fit_transform(x_train[:,7:])
            x_valid_pca = pca.transform(x_valid[:,7:])
            x_test_pca = pca.transform(x_test[:,7:])

            #Create all possibl e sequences
            for index in range(len(x_train_pca)-window_length):
                stock_windows_train.append(x_train_pca[index:index+window_length])

            x_train = np.array(stock_windows_train) #(X_inputs,window_size,indicators)


            for index in range(len(x_valid_pca)-window_length):
                stock_windows_train.append(x_valid_pca[index:index+window_length])

            x_valid = np.array(stock_windows_valid) #(X_inputs,window_size,indicators)


            for index in range(len(x_test_pca)-window_length):
                stock_windows_test.append(x_test_pca[index:index+window_length])

            x_test = np.array(stock_windows_test) #(X_inputs,window_size,indicators)

            #Assume close price adjusted in raw data, thus up to last value




            return [x_train,y_train,x_valid,y_valid,x_test,y_test]
































if __name__ == "__main__":
    print('Running stock predictor test')