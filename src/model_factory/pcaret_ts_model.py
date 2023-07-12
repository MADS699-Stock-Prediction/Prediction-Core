# can not install on the local system works on colab only
from pycaret.time_series import *
import pandas as pd


def get_best_ts_model(stock_id =['TSLA'], 
               start_date ='2008-03-01', 
               end_date ='2023-06-20', 
               clean_tech_data_store_dir='data/clean_data',
               model_storage_path = 'models/'):
    for id in stock_id:
        clean_file_path = clean_tech_data_store_dir + "/tech_indicator_pygooglenews" + id + "_"+start_date +"_" +end_date

    df = pd.read_csv(clean_file_path)
    df = df.asfreq('B')
    s = setup(data =df.Close , fh = 3, fold = 5, session_id = 123, numeric_imputation_target ='ffill')
    best = compare_models()
    plot_model(best, plot = 'forecast', data_kwargs = {'fh' : 24})
    plot_model(best, plot = 'diagnostics')
    plot_model(best, plot = 'insample')
    final_best = finalize_model(best)
    predict_model(best, fh = 24)
    with open(model_output_path, 'wb+') as out:
        pickle.dump(final_best, out)


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

    get_best_ts_model(stock_id=args.stock_ids, 
                      raw_tech_data_store_dir =args.raw_tech_data_store_dir,
                        model_storage_path = args.model_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)