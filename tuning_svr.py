from functions import *
from create_raw_data import *
from performance_metrics import *
import sklearn.preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import GridSearchCV,PredefinedSplit
#from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.model_selection import learning_curve
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from numpy.random import seed



seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

"""------------------------- Data processing -----------------------------"""

Returns = False
#Create dataset
df = create_finance(returns=False,plot_corr=False,Trends=False)

windows = [1, 3, 5, 7, 10, 15, 20]
for i in windows:
    seed(20)
    #Create sliding windows
    data = prepare_data(df,i,10,10,returns=False,normalize_cheat=False)
    data.create_windows()
    #data.normalized_windows()
    x_train = data.x_train.reshape(len(data.x_train),-1)
    x_valid = data.x_valid.reshape(len(data.x_valid),-1)
    x_test = data.x_test.reshape(len(data.x_test),-1)
    y_train = data.y_train.reshape(-1, 1)
    y_valid = data.y_valid.reshape(-1, 1)
    y_test = data.y_test.reshape(-1, 1)
    my_test_fold = []
    for j in range(len(x_train)):
        my_test_fold.append(-1)

    for k in range(j,j+len(x_valid)):
        my_test_fold.append(0)
    print(x_train.shape)
    x_train = np.concatenate((x_train,x_valid),axis=0)
    y_train = np.concatenate((y_train,y_valid),axis=0)
    print(x_train.shape)

    ps = PredefinedSplit(test_fold = my_test_fold)
 #   parameters = {'kernel': ('rbf','linear','poly'), 'C':[0.05,0.1,0.5, 1, 1.5, 10],'gamma': [100,10,10e-1,10e-2,10e-3,1e-7, 1e-4],'epsilon':[0.1,0.2,0.3]}
#    parameters = {'kernel': ['rbf'], 'C':np.linspace(4**-10,4**4,10),'gamma':np.linspace(4**-10,4**4,10),'epsilon':np.linspace(4**-10,4**-1,10)}
    parameters = {'kernel': ['rbf'], 'C':np.logspace(-8,3,12),'gamma':np.logspace(-8,3,12),'epsilon':np.logspace(-8,-1,8)}

    svr = svm.SVR()
    clf = GridSearchCV(svr,parameters,cv=ps,scoring='neg_mean_squared_error')
    clf.fit(x_train,y_train.ravel())
    print('<><><><><><><><><><><><><><><>')
    print('Window size: {}'.format(i))
    min = data.mm[0]
    max = data.mm[1]
    y_pred = clf.predict(x_test) * (max - min) + min
    y_test = y_test * (max - min) + min
    y_test = y_test.ravel()
    print('RMSE: {}'.format(np.sqrt(mse(y_test, y_pred))))
    print('RMSE2: {}'.format(np.sqrt(RMSE(y_test, y_pred))))
    print('MAPE: {}'.format(mean_absolute_percentage_error(y_test, y_pred)))
    print('MAPE2: {}'.format(MAPE(y_pred, y_test)))
    print('AMAPE: {}'.format(AMAPE(y_pred, y_test)))
    print('MAE: {}'.format(MAE(y_pred, y_test)))
    print('PCT: {}'.format(PCT(y_test, y_pred)))
    print('Window: {}'.format(i))
    print(clf.best_params_)
    print()
'''
data = prepare_data(df,1,10,10,returns=False,normalize_cheat=False)
data.create_windows()
#data.normalized_windows()
x_train = data.x_train.reshape(len(data.x_train),-1)
x_valid = data.x_valid.reshape(len(data.x_valid),-1)
x_test = data.x_test.reshape(len(data.x_test),-1)
y_train = data.y_train.reshape(-1, 1)
y_valid = data.y_valid.reshape(-1, 1)
y_test = data.y_test.reshape(-1, 1)

svr = svm.SVR(kernel='linear',gamma =1e-07,epsilon=0.1,C=1.5)
svr.fit(x_train,y_train.ravel())
#{'C': 1.5, 'epsilon': 0.1, 'gamma': 1e-07, 'kernel': 'linear'}

min = data.mm[0]
max = data.mm[1]
ps = Pre
y_pred = svr.predict(x_test) * (max - min) + min
y_test = y_test * (max - min) + min
y_test = y_test.ravel()

print('RMSE: {}'.format(np.sqrt(mse(y_test, y_pred))))
print('RMSE2: {}'.format(np.sqrt(RMSE(y_test, y_pred))))
print('MAPE: {}'.format(mean_absolute_percentage_error(y_test, y_pred)))
print('MAPE2: {}'.format(MAPE(y_pred, y_test)))
print('AMAPE: {}'.format(AMAPE(y_pred, y_test)))
print('MAE: {}'.format(MAE(y_pred, y_test)))
print('PCT: {}'.format(PCT(y_test, y_pred)))
'''