import numpy as np
import pandas as pd

from datetime import datetime
############################# UNIT  TEST ###############################################
""" Mean sentiment using Vader
        Date   score
0 2018-01-03  0.9922
1 2018-01-04  0.9922
2 2018-01-05  0.9922
3 2018-01-06  0.9922
4 2018-01-07  0.9922
########################################################################################
"""
def combine_eodnews_sentiment(tech_fundamental_combined, 
                        sentiment_df):
    """ 
    Returns combined(tech_fundamental +sentiment data) dataframe to the caller
    tech_fundamental_combined dataframe with technfundamental information
    sentiment_df the dataframe where all sentiment data are available
    """
    # uncomment below to test standalone

    #print(fundamentals_path, tech_indicator_df)
    tech_fundamental_combined.index = pd.to_datetime(tech_fundamental_combined.Date)
    #print(tech_indicator_df.index)
    sentiment_df.Date = pd.to_datetime(sentiment_df.Date)
    sentiment_df = sentiment_df.set_index('Date')
    merged_df = pd.merge(tech_fundamental_combined, sentiment_df, 
                         how='left', right_index=True, left_index=True)
    merged_df = merged_df.fillna(method='ffill').dropna()
    print(merged_df.head())
    return merged_df
