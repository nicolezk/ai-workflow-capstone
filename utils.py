import os
import re
import pandas as pd

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