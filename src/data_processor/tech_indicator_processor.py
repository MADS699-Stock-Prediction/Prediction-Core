import pandas as pd
import numpy as np
from numpy import log
import io
import pandas_ta as ta

def process_tech_indicator_data(stock_id =['TSLA'], start_date ='2008-03-01', 
                            end_date ='2023-06-20',
                            raw_tech_data_store_dir ='data/raw_data',
                            clean_tech_data_store_dir='data/clean_data'):
    for id in list(stock_id):
        print(id)
        raw_file_path = raw_tech_data_store_dir + "/tech_indicator_" + id + "_"+start_date +"_" +end_date
        clean_file_path = clean_tech_data_store_dir + "/tech_indicator_" + id + "_"+start_date +"_" +end_date
        df = pd.read_csv(raw_file_path)
        ##################Add all technical indicators#################
        append_tech_indicators(df)
        #################End technical indicators#######################
        df = df.dropna()
        df.to_csv(clean_file_path)
def append_tech_indicators(df):
        #get indicator lists
        indicators = df.ta.indicators(as_list=True)
        print("All available technical indicator list", indicators)
        df.ta.sma(50,append=True)
        df.ta.macd(close='close', fast=12, slow=26, append=True)
        df.ta.rsi(close='close',append=True)
        df.ta.tsi(close='close',append=True)

        df.ta.bbands(close='close', append=True)
        #volume
        df.ta.pvi( append=True)
        df.ta.nvi( append=True)
        df.ta.obv( append=True)
        # performance 
        df.ta.atr( append=True)
        df.ta.pvi( append=True)
        #Trend
        df.ta.adx(append=True)
        df.ta.increasing(append=True)    
        df.ta.decreasing(append=True)    
        df.ta.vwma(append=True)    
    
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
