# -----------------------------------------------------------
#
# Metrics to evaluate the model performance
#
# -----------------------------------------------------------

import numpy as np
import tensorflow.keras.backend as K

def metric_IA(y_true,y_pred):
    sse = K.sum(K.square(y_true - y_pred))
    ia = K.sum(K.square(K.abs(y_true - K.mean(y_true)) + K.abs(y_pred - K.mean(y_true))))

    ia = 1. - sse/ia
    return ia

def IA(true, predicted):  
    return 1 - np.sum((true-predicted)**2)/np.sum((np.abs(predicted-np.mean(true))+np.abs(true-np.mean(true)))**2)

def RMSE(true, predicted):
    return np.sqrt(np.mean((true-predicted)**2))

def NRMSE1(true, predicted):
    return np.sqrt(np.mean((true-predicted)**2))/(np.max(true)-np.min(true))

def NRMSE2(true, predicted):
    return np.sqrt(np.mean((true-predicted)**2))/np.mean(true)

def MAE(true, predicted):
    return np.mean(np.abs(true-predicted))

def get_metric_periodically(df, true, predicted, period, func):
    metric = []
    for t in range(int(df[period].max())+1):
        metric.append(func(df.loc[df[period] == t, true].values.ravel(), df.loc[df[period] == t, predicted].values.ravel()))
    return metric
    
def evaluate_forecast(df_na, true, predicted, period):
    result = dict()
    df = df_na.dropna(subset=[true, predicted]).copy()
    result['overall'] = {'IA': IA(df[true].values.ravel(), df[predicted].values.ravel()), 
                         'RMSE': RMSE(df[true].values.ravel(), df[predicted].values.ravel()), 
                         'NRMSE1': NRMSE1(df[true].values.ravel(), df[predicted].values.ravel()), 
                         'NRMSE2': NRMSE2(df[true].values.ravel(), df[predicted].values.ravel()),
                         'MAE': MAE(df[true].values.ravel(), df[predicted].values.ravel())}

    result[period] = {'IA': get_metric_periodically(df, true, predicted, period, IA),
                             'RMSE': get_metric_periodically(df, true, predicted, period, RMSE), 
                             'NRMSE1': get_metric_periodically(df, true, predicted, period, NRMSE1),
                             'NRMSE2': get_metric_periodically(df, true, predicted, period, NRMSE2),
                             'MAE': get_metric_periodically(df, true, predicted, period, MAE)}

    return result
