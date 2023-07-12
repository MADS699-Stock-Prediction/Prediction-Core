#!pip install yfinance
# TO-DO use the pygooglenews not used due to install issue
# This is just a dummy PoC to take a decision how to associate the 
# the news data with the tech indicators
# First go we can go with sentiment of the title itself
# Comments to be updated

#from GoogleNews import GoogleNews
from pygooglenews import GoogleNews
from gnews import GNews
import pandas as pd

#googlenews = GoogleNews()
googlenews = GoogleNews()
google_news = GNews()
google_news.start_date=(2022,1,1)
google_news.end_date=(2023,7,1)
def download_gnews_data(stock_id =['TSLA'], start_date ='02/01/2020', 
                            end_date ='02/01/2022',
                                 raw_tech_data_store_dir='data/raw_data/'):
    print(stock_id,raw_tech_data_store_dir)

    for id in list(stock_id):
        print(id)
        data = googlenews.search(id, when='2y')['entries']
        gnews_data =pd.DataFrame(data)
        file_path_pygnews = raw_tech_data_store_dir + "/googlenewsdata_pygnews_" + id + "_"+start_date +"_" +end_date
        file_path_gnews = raw_tech_data_store_dir + "/googlenewsdata_gnews_" + id + "_"+start_date +"_" +end_date
        print(gnews_data.published)
        gnews_data.to_csv(file_path_pygnews)
        data = pd.DataFrame(google_news.get_news(id))
        data.save(file_path_gnews)
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
    download_gnews_data(stock_id=args.stock_ids, raw_tech_data_store_dir =args.raw_tech_data_store_dir,
                            start_date = args.start_date,
                                end_date = args.end_date)