from datetime import date, datetime, timedelta
import pandas as pd
import urllib.request
from os.path import exists
import urllib, json

def get_datetime(datestr):
    from datetime import datetime
    date_format = "%Y-%m-%dT%H:%M:%S"
    date_string = datestr
    date_string = date_string.split('+')[0]
    # Convert string to datetime using strptime
    date_obj = datetime.strptime(date_string, date_format)
    return date_obj

def get_last_date_of_month(year, month):   
    if month == 12:
        last_date = datetime(year, month, 31)
    else:
        last_date = datetime(year, month + 1, 1) + timedelta(days=-1)    
    return last_date.strftime("%Y-%m-%d")

def combine_title_content(news_df):
    news_df['news_body'] = news_df['title'] + " " + news_df['content']
    news_df2 = news_df[['link', 'date_time', 'news_body']]
    return news_df2

def get_news_from_eodhistoricaldata(stock_id =['TSLA'], start_date ='02/01/2020', 
                            end_date ='02/01/2022',
                                 raw_tech_data_store_dir='../data/raw_data/eodnewsdata/'):
    print(stock_id,raw_tech_data_store_dir)

    for id in list(stock_id):
        print(id)
        file_path_eodnews = raw_tech_data_store_dir + "/eodnewsdata_" + id + "_"+start_date +"_" +end_date
        file_exists = exists(file_path_eodnews)
        if not file_exists:
            news_data = get_news_data(id, start_date,end_date,file_path_eodnews)
            news_data['date_time'] = news_data['date'].apply(get_datetime)
            news_data.to_csv(file_path_eodnews)


def get_news_data(tikr, start_date, end_date,file_path_eodnews):

    ticker=tikr
    api_token = ""
    file = file_path_eodnews+ ".json"
    
    file_exists = exists(file)
    
    result = list()
    
    if (file_exists):
        # JSON file
        f = open (file, "r")  
        # Reading from file
        result = json.loads(f.read())
    else:
        for year in range(2018,2024):
            for mon in range(1,13):
                frm = str(year) + "-" + "{:02d}".format(mon) + "-01"
                to = get_last_date_of_month(year, mon)
                file = "news_"+ str(year) + "-" + "{:02d}".format(mon)

                url = "https://eodhistoricaldata.com/api/news?api_token=" + api_token + "&s=" + ticker + \
            "&offset=0&from=" + frm + "&to=" + to + "&limit=1000&fmt=json"
                print(file + ": " + url)

                response = urllib.request.urlopen(url)
                data = json.loads(response.read())
                result.extend(data)
        print("I am DONEEEEEEEEEEEEEEEEEEEEEEEEEEEE!!!!!!!!!!!!!!!!")
    with open(file, "w") as outfile:
        json.dump(result, outfile)
    result_new = pd.json_normalize(result)    
    result_new['date_time'] = result_new['date'].apply(get_datetime)
    print(result_new.head())
    print(result_new.columns)
    #print(result_new[['sentiment.polarity','sentiment.pos','sentiment.neg','sentiment.neu','sentiment']])
    return result_new

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock_ids",
        '--names-list',
        nargs="*",  
        default=['TSLA'],
        )
    parser.add_argument('-raw_tech_data_store_dir', help='directory to store raw eod news data')
    parser.add_argument('-start_date', help='start date information')
    parser.add_argument('-end_date', help='end_date information')
    args = parser.parse_args()
    print(args)
    get_news_from_eodhistoricaldata(stock_id=args.stock_ids, raw_tech_data_store_dir =args.raw_tech_data_store_dir,
                            start_date = args.start_date,
                                end_date = args.end_date)