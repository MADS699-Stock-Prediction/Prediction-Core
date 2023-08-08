import pandas as pd
import numpy as np
from numpy import log
import io
import pandas_ta as ta
# Combine helper
from combine_tech_and_fundamentals import combine_tech_indicators_and_fundamentals
from combine_eodnews_sentiment import combine_eodnews_sentiment

def process_tech_indicator_data(stock_id =['TSLA'], start_date ='2008-03-01', 
                            end_date ='2023-06-20',
                            raw_tech_data_store_dir ='data/raw_data',
                            clean_tech_data_store_dir='data/clean_data',
                            eod_sentiment_data_path = 'data/clean_data/eodnewsdata'):
    for id in list(stock_id):
        print(id)
        raw_file_path = raw_tech_data_store_dir + "/tech_indicator_" + id + \
                                    "_"+start_date +"_" +end_date
        combined_clean_file_path = clean_tech_data_store_dir + "/all_combined" + \
                                        "/tech_fundamental_sentiment_" + id + \
                                     "_"+start_date +"_" +end_date
        fundamental_path = raw_tech_data_store_dir + "/tech_fundamental_" + id + ".csv"
                                    
        eod_sentiment_clean_path = clean_tech_data_store_dir + "/eodnewsdata" + "/eodnewsdata_" + id + \
                                     "_"+start_date +"_" +end_date
        df = pd.read_csv(raw_file_path)
        ##################Add all technical indicators#######################
        append_tech_indicators(df)
        #################End technical indicators############################

        ##################Combine fundamentals with the indicators###########
        if id =='TSLA':
            new_df = combine_tech_indicators_and_fundamentals( 
                                    fundamentals_path = fundamental_path, 
                                    tech_indicator_df = df)
        else:
            new_df = df
        ##################End combine fundamentals with tech indicators######
        ##################Combine sentiment with tech and fundamental########
        sentiment_df = pd.read_csv(eod_sentiment_clean_path)
        final_df = combine_eodnews_sentiment(new_df, sentiment_df)
        ##################Combine sentiment with tech and fundamental########

        df = final_df.dropna()
        df.to_csv(combined_clean_file_path)

def append_tech_indicators(df):
        #get indicator lists
        indicators = df.ta.indicators(as_list=True)
        df.ta.sma(50,append=True)
        df.ta.tsi(close='Adj close',append=True)

        df.ta.bbands(close='Adj close', append=True)
        #volume
        df.ta.pvi(close='Adj Close', append=True)
        df.ta.nvi(close='Adj Close', append=True)
        # performance 
        df.ta.atr(close='Adj Close', append=True)
        df.ta.pvi(close='Adj Close', append=True)
        #Trend
        df.ta.adx(close='Adj Close',append=True)
        df.ta.increasing(close='Adj Close', append=True)    
        df.ta.decreasing(close='Adj Close', append=True)    
        df.ta.vwma(close='Adj Close',append=True)    
        # Moving average convergence divergence
        df.ta.macd(close='Adj Close', fast=12, slow=26, append=True)
        df.ta.rsi(close='Adj Close',  append=True) # Relative strength index
        df.ta.obv(close='Adj Close', append=True) # On Balance Volume
        df.ta.ad(close='Adj Close', append=True) # Accumulation distribution line
        df.ta.adx(close='Adj Close', append=True) # Average directional index
        df.ta.ao(close='Adj Close', append=True) # Aroon oscillator
        return df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock_ids",
        '--names-list',
        nargs="*",  
        default=['TSLA'],
        )
    parser.add_argument('-raw_tech_data_store_dir', help='directory to store raw technical data files')
    parser.add_argument('-clean_tech_data_store_dir', help='directory to store clean technical data files')
    parser.add_argument('-start_date', help='start date information')
    parser.add_argument('-end_date', help='end_date information')
    args = parser.parse_args()    
    print(args)
    process_tech_indicator_data(stock_id=args.stock_ids, 
                                raw_tech_data_store_dir =args.raw_tech_data_store_dir,
                                clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                                start_date = args.start_date,
                                end_date = args.end_date)
