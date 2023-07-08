#!pip install yfinance
import pandas as pd
import numpy as np
from numpy import log
import io

import yfinance as yf
import requests


def download_tech_indicator(stock_id =['TSLA'], start_date ='2008-03-01', 
                            end_date ='2023-06-20',
                                 raw_tech_data_store_dir='data/raw_data/'):
    print(stock_id,raw_tech_data_store_dir)
    for id in list(stock_id):
        print(id)
        tech_indicator_data = yf.download(id,start_date,end_date)
        file_path = raw_tech_data_store_dir + "/tech_indicator_" + id + "_"+start_date +"_" +end_date
        print(file_path)
        tech_indicator_data.to_csv(file_path)


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
    parser.add_argument('-start_date', help='start date information')
    parser.add_argument('-end_date', help='end_date information')
    args = parser.parse_args()
    print(args)
    download_tech_indicator(stock_id=args.stock_ids, raw_tech_data_store_dir =args.raw_tech_data_store_dir,
                            start_date = args.start_date,
                                end_date = args.end_date)