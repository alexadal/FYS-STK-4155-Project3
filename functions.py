from pandas import read_csv
import numpy as np
from pandas import datetime
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()
with lstm_graph.as_default():


class prepare_data():
    def __init__(self,df,window_lenght,test_percent=20,Normalize = 'Price'):
        self.test_percent = test_percent
        self.data = df


    def normalize(self,Normalize):





    def create_windows(self,window_length):
        data_matrix = self.data.asmatrix()
        stock_windows = []

        #Create all possible sequences
        for index in range(len(data_matrix)-window_length):
            stock_windows.append(data_matrix[index:index+window_length])

        stock_window = np.array(stock_windows) #(X_inputs,window_size,indicators)
        test_size = int(np.round(self.test_percent/100*len(stock_windows.[0])))
        train_size = stock_windows.shape[0] - test_size

        #Assume close price adjusted in raw data
        x_train = stock_windows[:train_size, :, :-1]
        y_train = stock_windows[:train_size, :, :]

































if __name__ == "__main__":
    print('Running stock predictor test')