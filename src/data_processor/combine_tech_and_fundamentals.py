import numpy as np
import pandas as pd

import yfinance as yf
from datetime import datetime
############################# UNIT  TEST ###############################################
"""
	Adj Close	Volume	roe	roa	debt/asset	ev/ebitda	p/e
Date							
2020-03-31	34.933334	266572500	0.1744	0.04295	0.288054	31.530737	-576.457663
2020-04-01	32.104000	200298000	0.1744	0.04295	0.288054	28.976987	-529.768978
2020-04-02	30.298000	297876000	0.1744	0.04295	0.288054	27.346896	-499.967002
2020-04-03	32.000668	338431500	0.1744	0.04295	0.288054	28.883719	-528.063821
2020-04-06	34.416000	223527000	0.1744	0.04295	0.288054	31.063793	-567.920798

########################################################################################
"""
def combine_tech_indicators_and_fundamentals(fundamentals_path, 
                                             tech_indicator_df):
    """ 
    Returns combined(tech + fundamentl) dataframe to the caller
    fundamental_path file path where fundamentl is kept
    tech_indicator_df the dataframe where all tech indicators are available
    fundamental indicator data and logic from Nathan
    """
    # uncomment below to test standalone
    #start = datetime(2020,2,1)
    #end = datetime(2023,3,31)

    # Set the ticker
    #ticker = 'TSLA'

    # Get the data
    #data = yf.download(ticker, start, end)
    #data = data[['Adj Close', 'Volume']]
    # may be implace can be used to save memory later TODO
    #print(fundamentals_path, tech_indicator_df)
    tech_indicator_df.index = pd.to_datetime(tech_indicator_df.Date)
    #print(tech_indicator_df.index)
    stock_fundamentals = pd.read_csv(fundamentals_path, index_col=[0])
    stock_fundamentals.index = pd.to_datetime(stock_fundamentals.index)
    merged_df = pd.merge(tech_indicator_df, stock_fundamentals, 
                         how='left', right_index=True, left_index=True)
    merged_df = merged_df.fillna(method='ffill').dropna()

    # making fundamentals
    merged_df['debt/asset'] = merged_df['debt'] / merged_df['totalasset']
    merged_df['ev/ebitda'] = merged_df['Adj Close']*merged_df['shares'] / merged_df['ebitda']
    merged_df['p/e'] = merged_df['Adj Close'] / merged_df['eps']
    # drop unwanted columns as of now need to be double sure later TODO
    columns_to_drop =['bv','totalasset','eps','shares','ebitda','debt','dps']
    merged_df.drop(columns_to_drop, axis=1)
    #print(merged_df[['Adj Close', 'Volume', 'roe', 'roa', 'debt/asset', 'ev/ebitda', 'p/e']][310:330])
    return merged_df
