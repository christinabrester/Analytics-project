import numpy as np
import pandas as pd

def apply_interval_mask(df, start_date, end_date):
    mask = (df.loc[:, 'UTC'] >= start_date) & (df.loc[:, 'UTC'] <= end_date)
    df = df.loc[mask].reset_index(drop=True)
    return df

def select_columns(df, columns):
    df = df.loc[:, columns]
    return df

def get_cols_with_gaps(df):
    gaps = df.isna().sum(axis = 0)
    print("Columns with missing values in {}".format(df.name))
    print(gaps[gaps != 0])
    return list(gaps[gaps != 0].index)

def add_cos_sin_day(df):
    df['day_cos'] = df['day'].apply(lambda x: np.cos(x*2*np.pi/365.25))
    df['day_sin'] = df['day'].apply(lambda x: np.sin(x*2*np.pi/365.25))    
    return df


def add_cos_sin_hour(df):
    df['hour_cos'] = df['hour'].apply(lambda x: np.cos(x*2*np.pi/24))
    df['hour_sin'] = df['hour'].apply(lambda x: np.sin(x*2*np.pi/24))   
    return df

def preprocess(df, name, selected_columns, start_date, end_date):
    # Select data from the interval
    df = apply_interval_mask(df, start_date, end_date)
    print(df)

    # Keep the columns of interest data from the interval
    df = select_columns(df, selected_columns)

    # Define the df name 
    df.name = name

    # Print short info about gaps
    col_gaps = get_cols_with_gaps(df)

    # Fill gaps
    for col in col_gaps:
        df[col] = df[col].interpolate(method='polynomial', order=2).bfill(axis ='rows').ffill(axis ='rows')
        if col in ['power [W]', 'forecast fmi [W]', 's_glob', 's_dif', 'wind_avg', 'albedo', 'prec_amt',  'cl_tot', 'cl_low', 'cl_med', 'cl_high']:
            df[col] = [0 if value < 0 else value for value in df[col]]

    # Add a daily time input
    df['day'] = df['UTC'].apply(lambda x: x.timetuple().tm_yday)
    df = add_cos_sin_day(df)

    # Add a hourly time input
    df['hour'] = df['UTC'].apply(lambda x: x.hour)
    df = add_cos_sin_hour(df)

    # Add residuals for FMI forecast
    df['residual fmi [W]'] = df['power [W]'] - df['forecast fmi [W]']  

    return df
    
    
