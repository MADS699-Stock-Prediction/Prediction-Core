import pandas_ta as ta
import pandas as pd

def return_train_test_with_feature(df):
    # drop below mentioned column to ensure not too much impact due to the same
    # Model will select auto pre-processing
    cols_to_drop = ['Date','Date.1','Close','High','Low','Open','Adj Close','totalasset','shares','dps','eps','ebitda','bv']
    df_close = pd.DataFrame()
    df_close['CUMPCTRET_6'] = df['Close']
    df_close = df_close.pct_change(periods =6)
 
    df['CUMPCTRET_6'] = df_close.CUMPCTRET_6
    # shift to predict 3 days advance pct_return 
    df['CUMPCTRET_6'] = df['CUMPCTRET_6'].shift(-3)

    df_train = df[:int(len(df)*(0.95))]
    df_test = df[int(len(df)*(0.95)):]
    
    df_train = df_train.drop(cols_to_drop,axis=1).dropna()
    df_test = df_test.drop(cols_to_drop,axis=1).dropna()
    
    return df_train, df_test
