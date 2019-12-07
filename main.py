from functions import *
import sklearn.preprocessing

import pandas as pd
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.svm import SVR

from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam



warnings.simplefilter(action='ignore', category=FutureWarning)
# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/rawdata_finance.xlsx'
nanDict = {}
df = pd.read_excel(filename,)
df.rename(index=int, columns={"AdjClose": "AdjClose"}, inplace=True)

print(df.columns)
#Moving Average:
def MA(df,col,n,name = 'MA_'):
    #df: DataFrame
    #col: name of column to calculate moving average from
    #n number of samples to include in averaging window
    return pd.Series(df[col].rolling(n,min_periods=n).mean(),name=name,index=df.index)
#Exponential Moving Average:
def EMA(df,col,n,name = 'EMA_'):
    #df: DataFrame
    #col: name of column to calculate moving average from
    #n number of samples to include in averaging window
    return pd.Series(df[col].ewm(span=n, min_periods=n).mean(),name=name,index=df.index)

def highest(data,end,t):
    if(end < t):
        t = end
    return max(data[(end-t):end])

def lowest(data,end,t):
    if(end < t):
        t = end
    return min(data[(end-t):end])


#Calculate ACD and VPT, PROC and VROC finally OBV
nrows = df.shape[0]
ACD = [0]*nrows
VPT = [0]*nrows
PROC = [0]*nrows
VROC = [0]*nrows
OBV = [0]*nrows
WR = [0]*nrows
MOM = [0]*nrows
for i in df.index:
    if int(i) > 0:
        ACD[i] = ACD[i-1] + df.get_value(i,'Volume')*((df.get_value(i,'Close')-df.get_value(i,'Low'))-(df.get_value(i,'High')-df.get_value(i,'Close')))/(df.get_value(i,'High')-df.get_value(i,'Low'))
        VPT[i] = VPT[i-1] + df.get_value(i,'Volume')*(df.get_value(i,'Close')-df.get_value(i-1,'Close'))/df.get_value(i-1,'Close')
    if int(i) >= 12:
        PROC[i] = (df.get_value(i,'Close')-df.get_value(i-12,'Close'))/df.get_value(i-12,'Close')*100
        VROC[i] = (df.get_value(i, 'Volume') - df.get_value(i - 12, 'Volume')) / df.get_value(i - 12, 'Volume') * 100
        #        WR[i] =  (highest(df['Close'],i,14)-df.get_value(i,'Close'))/(highest(df['Close'],i,14)-lowest(df['Close'],i,14))

    if int(i) > 0 and df.get_value(i,'Close')>=df.get_value(i-1,'Close'):
        OBV[i]=OBV[i-1]+df.get_value(i,'Volume')
    elif int(i) > 0 and df.get_value(i,'Close')<df.get_value(i-1,'Close'):
        OBV[i]=OBV[i-1]-df.get_value(i,'Volume')
    if int(i) >= 14:
        WR[i] = (highest(df['High'], i, 14) - df.get_value(i, 'Close')) / (highest(df['High'], i, 14) - lowest(df['Low'], i, 14))
        MOM[i] = df.get_value(i,'Close')-df.get_value(i-14,'Close')
df['ACD'] = pd.Series(data=ACD,name="ACD",index=df.index)
df['VPT'] = pd.Series(data=VPT, name='VPT',index=df.index)

#MACD
df['MACD'] = pd.Series(EMA(df,"Close",n=12)-EMA(df,"Close",n=26),name='MACD',index=df.index)

#Chaikin oscillator
df['CHO'] = pd.Series(EMA(df,"AdjClose",n=3)-EMA(df,"AdjClose",n=10),name='CHO',index=df.index)

STO_k = [0]*nrows
for i in df.index:
    if int(i)>5:
        STO_k[i] = (df.get_value(i,'Close')-lowest(df['Close'],i,5))/(highest(df['Close'],i,5)-lowest(df['Close'],i,5))
    #WR[i] = (highest(df['Close'],i))
df['STO_k'] =pd.Series(STO_k,name='STO_k',index=df.index)

#Stochastic oscilator D
df['STO_d'] = MA(df,col='STO_k',n=3,name='STO_d')
df['PROC'] = pd.Series(PROC,name='PROC',index=df.index)
df['VROC'] = pd.Series(VROC,name='VROC',index=df.index)
df['OBV'] = pd.Series(OBV,name='OBV',index=df.index)
df['WR'] = pd.Series(WR,name='WR',index=df.index)
print(df.columns)

# Missing Rsi as RS is not well defined (need to find out range)

#Relative Strength Index
# Taken from
# https://github.com/meenmo/Forecasting_Stock_Returns_via_Supervised_Learning/blob/master/Code/technical_indicators.py?fbclid=IwAR3S9zbc4fnQoo9pKg27AH8D819nVsn-WXL78D0Xpwzzu5HBdrEQD22MuME
i = 0
UpI = [0]
DoI = [0]
n = 14
while i + 1 <= df.index[-1]:
    UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
    DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
    if UpMove > DoMove and UpMove > 0:
        UpD = UpMove
    else:
        UpD = 0
    UpI.append(UpD)
    if DoMove > UpMove and DoMove > 0:
        DoD = DoMove
    else:
        DoD = 0
    DoI.append(DoD)
    i = i + 1
UpI = pd.Series(UpI)
DoI = pd.Series(DoI)
PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
df['RSI'] = pd.Series(PosDI / (PosDI + NegDI), name='RSI',index=df.index)
df['MOM'] = pd.Series(MOM,name='MOM',index=df.index)

#MedianPricePerday
df['MedianPrice'] = pd.Series((df['High']+df['Low'])/2,name='MedianPrice',index=df.index)

df['AO'] = pd.Series(MA(df,'MedianPrice',5)-MA(df,'MedianPrice',34),name='AO',index=df.index)
df['AC'] = pd.Series(df['AO']-MA(df,'AO',5))
df = df.drop(columns=['AO','MedianPrice'])

df = df.loc[37:]


cols = list(df.columns.values)
cols.pop(cols.index("Next_day"))
df = df[cols+["Next_day"]]
df = df.drop(columns = ["Index","Date"])
data = prepare_data(df,20,20,10)

data.create_windows(50)


#Not for LSTM
x_train = data.x_train.reshape(len(data.x_train),-1)
x_valid = data.x_valid.reshape(len(data.x_valid),-1)
x_test = data.x_test.reshape(len(data.x_test),-1)

svr = SVR(C=100,verbose=True)

svr.fit(x_train,data.y_train.ravel())

print(svr.score(x_test,data.y_test.ravel()))


i = Input(shape=(50, 8))
x = LSTM(5)(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(lr=0.1),
)

# train the RNN
r = model.fit(
  data.x_train, data.y_train.ravel(),
  epochs=100,
  validation_data=(data.x_valid,data.y_valid.ravel()),
)




plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.figure()
outputs = model.predict(data.x_test)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(data.y_test.ravel(), label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()

plt.figure()
outputs = model.predict(data.x_train)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(data.y_train.ravel(), label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()