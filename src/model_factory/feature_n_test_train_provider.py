import pandas_ta as ta
import pandas as pd
import numpy as np

def return_train_test_with_feature(df,stock_tickr):
    # drop below mentioned column to ensure not too much impact due to the same
    # Model will select auto pre-processing
    cols_to_drop = ['Unnamed: 0','Date','Date.1','Close','High','Low','Open','Adj Close','bv','totalasset','roe','roa','eps','shares','ebitda','debt','dps','debt/asset','ev/ebitda','p/e','Unnamed: 0' ]
    #cols_to_drop = 'bv','totalasset','roe','roa','eps','shares','ebitda','debt','dps','debt/asset','ev/ebitda','p/e','Unnamed: 0'
    if stock_tickr !='TSLA':
        cols_to_drop = ['Unnamed: 0','Date','Date.1','Close','High','Low','Open','Adj Close']
    df_close = pd.DataFrame()
    #df['log_ret'] = np.log(df.Close) - np.log(df.Close.shift(1))
    df['log_ret'] = df['Close']#.shift(-1)
    print(df.log_ret[:10])
    #df['log_ret'] =  df.Close.pct_change(periods=3)
    df['log_ret'] = df['log_ret'].shift(-3)


    df = df.dropna()
    df_train = df[:int(len(df)*(0.95))]
    df_test = df[int(len(df)*(0.95)):]
    
    df_train = df_train.drop(cols_to_drop,axis=1).dropna()
    df_test = df_test.drop(cols_to_drop,axis=1).dropna()
    
    return df_train, df_test

def return_train_test_with_feature_for_evaluate(df,stock_tickr):
    # drop below mentioned column to ensure not too much impact due to the same
    # Model will select auto pre-processing
    cols_to_drop = ['Unnamed: 0','Date','Date.1','Close','High','Low','Open','Adj Close','bv','totalasset','roe','roa','eps','shares','ebitda','debt','dps','debt/asset','ev/ebitda','p/e','Unnamed: 0' ]
    #cols_to_drop = 'bv','totalasset','roe','roa','eps','shares','ebitda','debt','dps','debt/asset','ev/ebitda','p/e','Unnamed: 0'
    if stock_tickr !='TSLA':
        cols_to_drop = ['Unnamed: 0','Date','Date.1','Close','High','Low','Open','Adj Close']
    df_close = pd.DataFrame()
    df['log_ret'] = df['Close'] # np.log(df.Close) - np.log(df.Close.shift(1))
    df['log_ret'] = df['log_ret'].shift(-3)
    df = df.dropna()
    df_train = df[:int(len(df)*(0.90))]
    df_test = df[int(len(df)*(0.9)):]
    
    df_train = df_train.drop(cols_to_drop,axis=1).dropna()
    df_test = df_test.drop(cols_to_drop,axis=1).dropna()
    
    return df_train, df_test
