#--------------------------------------------
#
# Auxiliary functions for the study 
#
#--------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def get_cv_splits(sample_size, n_folds, period):
    # sample_size with hourly time resolution
    # period/n_folds defines the number of days in each period from one fold; period in days
    cv_splits = []
    n_days = int(period/n_folds)
    while len(cv_splits) <= sample_size:
        for fold in range(n_folds):
            cv_splits.extend([fold]*24*n_days)

    return cv_splits[:sample_size]

def fit_scalers(df, inputs, output, test_id, run):
    mask = np.array(test_id) != run
    scaler_x = MinMaxScaler().fit(df.loc[mask, inputs])
    scaler_y = MinMaxScaler().fit(df.loc[mask, [output]])

    return scaler_x, scaler_y

def apply_scalers(df_, inputs, output, scaler_x, scaler_y):
    df = df_.copy()
    df.loc[:, inputs] = scaler_x.transform(df.loc[:, inputs])
    df.loc[:, [output]] = scaler_y.transform(df.loc[:, [output]])
    return df

def inverse_scale(df, scaler):
    return scaler.inverse_transform(df)
    
def generate_samples_with_lookback(df_, inputs, output, lookback):
    columns = inputs + [output]
    df = df_.loc[:, columns].copy()
    df = df.append(pd.Series([np.nan]*len(df.columns), index = df.columns), ignore_index = True)
    df[output] = df[output].shift(1)

    data_x = df.loc[:, inputs].values
    data_y = df.loc[:, output].values.ravel()
    
    data_gen = TimeseriesGenerator(data_x, data_y, batch_size = data_x.shape[0]-lookback, length=lookback)
    samples_x, samples_y = data_gen[0]
    return samples_x, samples_y

def get_train_and_test(data, test_id, run):
    mask_train = np.array(test_id) != run
    mask_test = np.array(test_id) == run
    return data[mask_train], data[mask_test]
    
def generate_samples_with_shift(df_, inputs, output, lookback):

    df = pd.DataFrame()
    for col in inputs:
        for i in range(lookback):
            df['{}-{}'.format(col, i)] = df_[col].shift(i)

    new_inputs = df.columns
    df = df.dropna(axis = 0)
    samples_x = df.values
    samples_y = df_[output].values[lookback-1:].reshape(-1,1)
    return samples_x, samples_y, new_inputs

    

    
