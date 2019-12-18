import tensorflow as tf
import keras
from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Dropout, Activation
from keras.models import Model, Sequential
from keras.activations import relu, tanh
import numpy as np
from performance_metrics import *
import talos as ta
from talos.utils import live
from functions import *
from create_raw_data import *
from talos.utils import lr_normalizer, best_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from sklearn.model_selection import PredefinedSplit
from sklearn import svm

from functions import mse
from keras.optimizers import SGD, Adam, Nadam
from keras.utils import plot_model


#Prepare SVM data with Google Trends
df = create_finance(returns=False,plot_corr=False,Trends=True)
seed(20)
#Create sliding windows
data = prepare_data(df,20,10,10,returns=False,normalize_cheat=False)
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
x_train = np.concatenate((x_train,x_valid),axis=0)
y_train = np.concatenate((y_train,y_valid),axis=0)
ps = PredefinedSplit(test_fold = my_test_fold)
svr = svm.SVR(C=1000,epsilon=0.01,gamma=0.0001,kernel='rbf')
svr.fit(X=x_train,y=y_train.ravel())
min = data.mm[0]
max = data.mm[1]

y_svr_trend = svr.predict(x_test) * (max - min) + min


#Prepare SVM data without Google Trends

df = create_finance(returns=False,plot_corr=False,Trends=False)
seed(20)
#Create sliding windows
data = prepare_data(df,5,10,10,returns=False,normalize_cheat=False)
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
x_train = np.concatenate((x_train,x_valid),axis=0)
y_train = np.concatenate((y_train,y_valid),axis=0)
ps = PredefinedSplit(test_fold = my_test_fold)
svr = svm.SVR(C=1000,epsilon=0.01,gamma=0.001,kernel='rbf')
svr.fit(X=x_train,y=y_train.ravel())
min = data.mm[0]
max = data.mm[1]

y_svr_notrend = svr.predict(x_test) * (max - min) + min



#Data from FFNN without Google Trendd
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    WithTrend = ta.Restore('Talos_Results/ANN_V2_stock_window_20_deploy.zip');
df = create_finance(returns=False, plot_corr=False, Trends=True)
data = prepare_data(df, 20, 10, 10, returns=False, normalize_cheat=False)
data.create_windows()
x_test = data.x_test.reshape(len(data.x_test), -1)
y_test = data.y_test.reshape(-1, 1)
min = data.mm[0]
max = data.mm[1]
y_ffnn_trend = WithTrend.model.predict(x_test) * (max - min) + min
y_test = y_test * (max - min) + min

#Data from FFNN with Google Trendd
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    WithoutTrend = ta.Restore('Talos_Results/ANN_WithoutTrends_stock_window_20_deploy.zip');

df = create_finance(returns=False, plot_corr=False, Trends=False)
data = prepare_data(df, 20, 10, 10, returns=False, normalize_cheat=False)
data.create_windows()
x_test = data.x_test.reshape(len(data.x_test), -1)

min = data.mm[0]
max = data.mm[1]
y_ffnn_notrend = WithoutTrend.model.predict(x_test) * (max - min) + min

#Data from lstm models
y_lstm_trend = [2938.4282, 2946.491, 2923.1594, 2910.2307, 2931.0215, 2921.8694, 2883.5586, 2882.3518, 2856.5793, 2871.6704, 2816.8972, 2841.1987, 2846.6313, 2875.5642, 2869.4312, 2856.152, 2870.365, 2865.7488, 2828.2727, 2830.6694, 2808.6516, 2783.7222, 2803.2212, 2756.68, 2762.6377, 2803.8003, 2827.823, 2851.6174, 2886.4614, 2913.8562, 2919.6948, 2902.0396, 2910.048, 2899.766, 2894.4653, 2930.6199, 2936.4421, 2968.3901, 2971.1477, 2947.8376, 2925.4524, 2917.9424, 2913.069, 2922.9, 2949.1387, 2965.8562, 2988.8984, 3000.1436, 2987.7388, 2978.1875, 2990.598, 2987.4382, 3007.4653, 3004.4922, 3009.41, 2994.9512, 2982.3516, 2967.8843, 2961.3928, 2984.6953, 3006.1382, 3006.3518, 3023.951, 3020.2046, 3008.9236, 2974.943, 2941.5984, 2912.465, 2839.6687, 2878.0608, 2863.1108, 2914.0544, 2917.718, 2904.7961, 2933.149, 2873.8284, 2866.7434, 2887.92, 2920.5125, 2907.6323, 2945.01, 2936.9995, 2893.0737, 2881.2263, 2871.7954, 2875.8457, 2913.197, 2940.933, 2913.928, 2960.2622, 2991.6287, 2986.215, 2985.833, 2985.8616, 3002.8318, 3012.7715, 3027.2776, 3006.7217, 3009.4546, 3001.026, 3003.539, 2991.461, 2977.9033, 2970.0867, 2966.2122, 2964.211, 2949.6082, 2969.8743, 2937.3257, 2895.485, 2907.9385, 2931.979, 2936.6672, 2913.0107, 2930.427, 2945.7502, 2975.42, 2977.5005, 3000.7585, 2993.7708, 3014.6143, 2997.6184, 3008.5178, 3003.9763, 3004.9553, 3011.639, 3020.1729, 3049.2395, 3043.3857, 3047.45, 3035.921, 3063.7363, 3071.2776, 3075.4622, 3068.9165, 3085.8125, 3076.3896, 3075.585, 3078.7144, 3079.5835, 3075.257, 3093.4526, 3103.8103, 3105.6216, 3094.4917, 3078.0105, 3083.9316, 3102.936, 3116.2173, 3121.7476, 3120.1953]
y_lstm_notrend = [2934.9094, 2940.0457, 2923.3218, 2912.0845, 2929.2803, 2913.5981, 2875.5198, 2872.7788, 2854.449, 2863.7444, 2807.0889, 2832.4133, 2843.1343, 2870.2534, 2864.8022, 2850.3284, 2866.394, 2858.4417, 2823.3765, 2824.9912, 2808.0637, 2780.424, 2799.9578, 2753.646, 2758.0623, 2796.9963, 2820.5256, 2841.1467, 2873.7354, 2897.3535, 2898.06, 2884.6729, 2898.0107, 2884.453, 2885.174, 2914.171, 2920.9875, 2951.765, 2958.745, 2937.3645, 2918.687, 2905.5654, 2903.6929, 2920.0671, 2935.0488, 2954.5183, 2976.3828, 2983.3198, 2976.1636, 2966.1663, 2976.3542, 2972.2126, 2990.8677, 2997.49, 2997.96, 2981.386, 2973.2812, 2959.1323, 2955.5537, 2982.9448, 2998.4014, 2995.1934, 3018.2366, 3013.563, 3001.2673, 2966.4375, 2938.062, 2909.6702, 2832.6055, 2871.3464, 2857.3499, 2911.168, 2907.884, 2895.3606, 2922.0886, 2857.7007, 2852.4695, 2881.929, 2910.8591, 2899.7944, 2932.9573, 2927.567, 2877.264, 2871.9192, 2862.7056, 2866.6206, 2905.0073, 2931.5894, 2908.387, 2944.8904, 2972.6401, 2969.8135, 2971.3538, 2974.3462, 2995.241, 2995.6895, 3011.522, 2993.5889, 2995.172, 2991.8318, 2995.2227, 2985.9087, 2969.5103, 2955.2014, 2955.9783, 2953.8662, 2941.5015, 2957.567, 2927.9668, 2882.627, 2899.3682, 2929.4033, 2925.486, 2901.3435, 2917.959, 2940.6145, 2964.3052, 2964.2634, 2990.2932, 2982.948, 2999.5513, 2987.3677, 2998.7473, 2992.8416, 2994.2764, 3003.36, 3009.9934, 3036.9988, 3030.5615, 3036.7605, 3026.4097, 3049.6965, 3055.822, 3058.8328, 3055.3518, 3068.15, 3061.8481, 3061.425, 3067.161, 3064.7983, 3065.8923, 3083.2163, 3096.095, 3095.6902, 3086.4495, 3072.5955, 3074.9338, 3095.5854, 3105.7603, 3114.1824, 3114.6726]


plt.figure()
plt.title('With Google Trend Data')
plt.plot(y_test, label='Targets')
#plt.plot(y_pred_trend[10:,0], label='predictions_with_trend')
plt.plot(y_ffnn_trend, label='FFNN (20 day windows)',linewidth=0.75)
plt.plot(y_lstm_trend, label='LSTM (20 day windows)',linewidth=0.75)
plt.plot(y_svr_trend, label='SVM (20 day windows)',linewidth=0.75)
plt.xlabel('DAYS')
plt.ylabel('S&P500')
plt.legend()
plt.show()



plt.figure()
plt.title('Without Google Trend Data')
plt.plot(y_test, label='Targets')
#plt.plot(y_pred_trend[10:,0], label='predictions_with_trend')
plt.plot(y_ffnn_notrend, label='FFNN (20 day windows)',linewidth=0.75)
plt.plot(y_lstm_notrend, label='LSTM (20 day windows)',linewidth=0.75)
#5 day windows means the Test data is 15 days longer than the version for 20 days
#Adjusting for that by not plotting the 15 first days of the SVR data
plt.plot(y_svr_notrend[15:], label='SVM (5 day windows)',linewidth=0.75)
plt.xlabel('DAYS')
plt.ylabel('S&P500')
plt.legend()
plt.show()

