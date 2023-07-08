import pandas as pd
import numpy as np
from numpy import log
import io

def process_tech_indicator_data(stock_id =['TSLA'], start_date ='2008-03-01', 
                            end_date ='2023-06-20',
                            raw_tech_data_store_dir ='data/raw_data',
                            clean_tech_data_store_dir='data/clean_data'):
    for id in list(stock_id):
        print(id)
        raw_file_path = raw_tech_data_store_dir + "/tech_indicator_" + id + "_"+start_date +"_" +end_date
        clean_file_path = clean_tech_data_store_dir + "/tech_indicator_" + id + "_"+start_date +"_" +end_date
        df = pd.read_csv(raw_file_path)
        df = df.dropna()
        df.to_csv(clean_file_path)

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
