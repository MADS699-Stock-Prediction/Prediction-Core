#pip install -q transformers
from transformers import pipeline
import pandas as pd

def senttiment_to_number(x):
    if x =='NEU':
        return 0
    if x =='POS':
        return 1
    else:
        return -1
    
def extract_gnews_sentiment(stock_id =['TSLA'], start_date ='02/01/2020', 
                            end_date ='02/01/2022',
                                 raw_tech_data_store_dir='data/googlenewsdata/'):
    print(stock_id,raw_tech_data_store_dir)

    for id in list(stock_id):
        print(id)
        gnews_data =pd.DataFrame(data)
        file_path_pygnews = raw_tech_data_store_dir + "/googlenewsdata_pygnews_" + id + "_"+start_date +"_" +end_date
        file_path_gnews = raw_tech_data_store_dir + "/googlenewsdata_gnews_" + id + "_"+start_date +"_" +end_date
# as of now sentiment from pygooglenews only
        sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
        df = pd.read_csv(file_path_pygnews)
        data = list(df.title)
        print(len(data))
        df['sentiment'] = pd.DataFrame(sentiment_pipeline(data))['label']
        df['sentiment'] = df['sentiment'].apply(senttiment_to_number)
        print(df.tail(10))
        df.to_csv(file_path_pygnews)


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
    parser.add_argument('-start_date', help='start date information')
    parser.add_argument('-end_date', help='end_date information')
    args = parser.parse_args()
    print(args)
    extract_gnews_sentiment(stock_id=args.stock_ids, raw_tech_data_store_dir =args.raw_tech_data_store_dir,
                            start_date = args.start_date,
                                end_date = args.end_date)