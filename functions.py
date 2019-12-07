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
        stock_window = []

        #Create all possible sequences
        for index in range(len(data_matrix)-window_length):
            stock_window.append(data_matrix[index:index+window_length])

        stock_window = np.array(stock_window) #(X_inputs,window_size,indicators)
        test_size = int(np.round(self.test_percent/100*stock_window.shape[0]))
        train_size = stock_window.shape[0] - test_size



































if __name__ == "__main__":
    print('Running stock predictor test')