import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
nltk.download('vader_lexicon')
from os.path import exists
import urllib, json

wsb_lexicon = {
    'moon': 3,
    'buy': 3,
    'buying': 3,
    'long': 3,
    'tendies': 3,
    'diamond': 3,
    'btfd': 3,
    'dd': 3,
    'pump': 3,
    'rocket': 3,
    'lambo': 3,
    'locked': 3,
    'loaded': 3,
    'rebound': 1,
    'massive': 1,
    'hawkish': 2,
    'citadel': 2,
    'bounce': 2,
    'hold': 2,
    'holding': 2,
    'call': 3,
    'calls': 3,
    'squeeze': 3,
    'gain': 2,
    'gains': 2,
    'liquidate': -3,
    'liquidated': -3,
    'put': -3,
    'puts': -3,
    'bagholder': -3,
    'bagholders': -3,
    'short': -3,
    'shorts': -3,
    'sell': -3,
    'paper': -3,
    'dump': -3,
    'yolo': 2,
    'guh':-3,
    '🐂': 3,
    '🍗': 3,
    '🚀🌝': 3,
    '💎':3,
    '😂': 3,
    '💎🤲':3,
    '🚀': 3,
    '🧸': -3,
    '🧻🤲': -3,
    '😢': -3
}

# Initialise vader sentiment analyser
vader_sia = SentimentIntensityAnalyzer()
vader_sia.lexicon.update(wsb_lexicon)

def get_mean_scores(df, group_by_col):
    print(group_by_col)
    # Group by date and calculate the mean
    mean_scores = df.groupby(pd.Grouper(key=group_by_col, axis=0, 
                          freq='1D', sort=True)).mean(numeric_only=True)

    # Unstack the mean_scores 
    mean_scores = mean_scores.unstack()

    # Get the cross-section of compound sentiment score
    mean_scores = mean_scores.xs('compound').transpose()
    
    res_df= mean_scores.to_frame().reset_index()
    res_df = res_df.rename(columns= {group_by_col: 'Date', 0: 'score'})
    res_df = res_df.ffill()
    return res_df

def combine_title_content(news_df):
    news_df['news_body'] = news_df['title'] + " " + news_df['content']
    news_df2 = news_df[['link', 'date_time', 'news_body']]
        # Iterate through the news and get the polarity scores using vader
    news_scores = news_df2['news_body'].apply(vader_sia.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    news_scores_df = pd.DataFrame(news_scores)

    # Join the DataFrames of the news and the list of dicts
    news_df2 = news_df2.join(news_scores_df, rsuffix='_right')
    return news_df2

def combine_eodnews_sentiment(stock_id= ['TSLA'], 
                            raw_tech_data_store_dir="../data/raw_data/eodnewsdata/",
                            clean_tech_data_store_dir= "../data/clean_data/eodnewsdata/",
                            start_date ='2018-06-20' ,
                                end_date= '2023-06-20'):
        
        for id in list(stock_id):
            print(id)
            #gnews_data =pd.DataFrame(data)
            file_path_eodnews = raw_tech_data_store_dir + "/eodnewsdata_" + id + "_"+start_date +"_" +end_date
            file_path_eodnews_with_sentiment = clean_tech_data_store_dir + "/eodnewsdata_" + id + "_"+start_date +"_" +end_date
            file_exists = exists(file_path_eodnews_with_sentiment)
            if not file_exists:
                print(file_path_eodnews,file_path_eodnews_with_sentiment)
                df = pd.read_csv(file_path_eodnews)
                df_combined = combine_title_content(df)
                df_combined['date_time'] = pd.to_datetime(df_combined['date_time'])
                print(df_combined.info(), df_combined.describe())
                score_df = get_mean_scores(df_combined,'date_time')
                score_df.to_csv(file_path_eodnews_with_sentiment)
                print(score_df.head())

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock_ids",
        '--names-list',
        nargs="*",  
        default=['TSLA'],
        )
    parser.add_argument('-raw_tech_data_store_dir', help='directory to append sentiment with raw data')
    parser.add_argument('-clean_tech_data_store_dir', help='directory to store clean technical data files')

    parser.add_argument('-start_date', help='start date information')
    parser.add_argument('-end_date', help='end_date information')
    args = parser.parse_args()
    print(args)
    combine_eodnews_sentiment(stock_id=args.stock_ids, 
                            raw_tech_data_store_dir =args.raw_tech_data_store_dir,
                            clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                            start_date = args.start_date,
                                end_date = args.end_date)