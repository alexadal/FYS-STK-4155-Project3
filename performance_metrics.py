
import numpy as np

def MAE(y_pred,y_true):
    if(len(y_pred) != len(y_true)):
        print('Inputs are not of equal length.')
        return 0
    n = len(y_pred)
    return (sum(abs(y_pred-y_true)))/n

def RMSE(y_pred,y_true):
    if(len(y_pred) != len(y_true)):
        print('Inputs are not of equal length.')
        return 0
    n = len(y_pred)
    return (sum((y_pred-y_true)**2))/n

def MAPE(y_pred,y_true):
    if(len(y_pred) != len(y_true)):
        print('Inputs are not of equal length.')
        return 0
    n = len(y_pred)
    return 100*(sum(abs((y_pred-y_true)/y_true))/n)

def AMAPE(y_pred,y_true):
    if(len(y_pred) != len(y_true)):
        print('Inputs are not of equal length.')
        return 0
    n = len(y_pred)
    mean_true = np.mean(y_true)
    return 100*(sum(abs((y_pred-y_true)/y_true))/n)

def PCT(y_pred,y_true):
    #Percentage Correct Trend
    pred = [0]*
    for i in range(len(y_pred)):
        if y_true