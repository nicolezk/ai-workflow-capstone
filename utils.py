import os
import re
import pandas as pd
from collections import defaultdict
from datetime import datetime
import numpy as np

def load_json_from_dir(dir):
    """
    Reads multiple json files in a directory, aggregates them, and sorts them by date.

    Parameters
    ----------
    dir: directory where the json files are

    Returns
    -------
    df: pandas DataFrame with datetimeindex
    """

    ## argument check
    if not os.path.isdir(dir):
        raise Exception('Input specified is not a valid directory')
    elif not len(os.listdir(dir)) > 0:
        raise Exception('Specified directory does not contain files')

    file_list = [os.path.join(dir, f) for f in os.listdir(dir) if re.search('\.json', f)]

    if not len(file_list) > 0:
        raise Exception('Specified directory does not contain json files')

    ## format columns and read files
    sorted_columns = ['country', 'customer_id', 'day', 'invoice', 'month',
                      'price', 'stream_id', 'times_viewed', 'year']
    
    df = None

    for f in file_list:
        _df = pd.read_json(f)

        ## common column typos
        _df = _df.rename(columns={'StreamID':'stream_id',
                                  'TimesViewed':'times_viewed',
                                  'total_price':'price'})

        if sorted(list(_df.columns)) != sorted_columns:
            raise Exception(f'Columns of file {f} could not be matched to the template \
                              (file columns: {list(_df.columns)})')

        _df['date'] = pd.to_datetime(_df[['year','month','day']])
        _df = _df.drop(columns=['year','month','day'])
        _df['invoice'] = _df['invoice'].str.replace(r'\D', '') # removes non digit characters from invoice id

        df = pd.concat([df,_df])

    df = df.set_index(pd.DatetimeIndex(df['date']))
    df = df.drop(columns=['date'])
    df = df.sort_index()

    return df

def aggregate_data(df):
    """
    Obtains summarized data (number of purchases, number of unique invoices, 
    number of unique streams, total number of views, and total revenue) per day
    (grouped by country)

    Parameters
    ----------
    df: pandas DataFrame

    Returns
    -------
    summ_df: pandas DataFrame
    """
    summ_df = df.groupby(['country','date']).agg(
                                    purchases=('customer_id','count'),
                                    unique_invoices=('invoice','nunique'),
                                    unique_streams=('stream_id','nunique'),
                                    total_views=('times_viewed','sum'),
                                    revenue=('price','sum')).reset_index('country')

    return summ_df

def engineer_features(df,training=True):
    """
    for any given day the target becomes the sum of the next days revenue
    for that day we engineer several features that help predict the summed revenue
    
    the 'training' flag will trim data that should not be used for training
    when set to false all data will be returned
    """

    ## extract dates
    df = df.reset_index().copy()
    dates = df['date'].values.copy()
    dates = dates.astype('datetime64[D]')

    ## engineer some features
    eng_features = defaultdict(list)
    previous =[7, 14, 28, 70]  #[7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    y = np.zeros(dates.size)
    for d,day in enumerate(dates):

        ## use windows in time back from a specific date
        ## calculate sum of previous revenue for past 7 days (and other ranges)
        for num in previous:
            current = np.datetime64(day, 'D') 
            prev = current - np.timedelta64(num, 'D')
            mask = np.in1d(dates, np.arange(prev,current,dtype='datetime64[D]'))
            eng_features["previous_{}".format(num)].append(df[mask]['revenue'].sum())

        ## get the target revenue    
        plus_30 = current + np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(current,plus_30,dtype='datetime64[D]'))
        y[d] = df[mask]['revenue'].sum()

        ## attempt to capture monthly trend with previous years data (if present)
        # calculate sum of revenue of target month from previous year
        start_date = current - np.timedelta64(365,'D')
        stop_date = plus_30 - np.timedelta64(365,'D')
        mask = np.in1d(dates, np.arange(start_date,stop_date,dtype='datetime64[D]'))
        eng_features['previous_year'].append(df[mask]['revenue'].sum())

        ## add some non-revenue features
        minus_30 = current - np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(minus_30,current,dtype='datetime64[D]'))
        eng_features['recent_invoices'].append(df[mask]['unique_invoices'].mean())
        eng_features['recent_views'].append(df[mask]['total_views'].mean())

    X = pd.DataFrame(eng_features)
    ## combine features in to df and remove rows with all zeros
    X.fillna(0,inplace=True)
    mask = X.sum(axis=1)>0
    X = X[mask]
    y = y[mask]
    dates = dates[mask]
    X.reset_index(drop=True, inplace=True)

    if training == True:
        ## remove the last 30 days (because the target is not reliable)
        mask = np.arange(X.shape[0]) < np.arange(X.shape[0])[-30]
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        X.reset_index(drop=True, inplace=True)
    
    return(X,y,dates)

data_dir = '0_exercise_files/cs-train/'
df = load_json_from_dir(data_dir)
summ_df = aggregate_data(df)
print(summ_df.head(5))
X = engineer_features(summ_df)