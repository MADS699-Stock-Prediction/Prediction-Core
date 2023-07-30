# can not install on the local system works on colab only
from pycaret.time_series import *
from pycaret import *
import pandas as pd
import pickle
import numpy as np

def get_best_ts_model(stock_id =['TSLA'], 
               start_date ='2008-03-01', 
               end_date ='2023-06-20', 
               clean_tech_data_store_dir='data/clean_data',
               model_storage_path = 'models/'):
    for id in stock_id:
        clean_file_path = clean_tech_data_store_dir + "/tech_indicator_" + id + "_"+start_date +"_" +end_date
    print(clean_file_path)
    df = pd.read_csv(clean_file_path)
    df = df[['Close']]
    print(df.head())
    #df = df.asfreq('B')
    print(df.head())
    df = df[['Close']]
    print(df.head())
    df['log_ret'] = np.log(df.Close) - np.log(df.Close.shift(1))

    s = setup(data =df.log_ret , fh = 3, fold = 5, session_id = 123, numeric_imputation_target ='ffill', log_experiment=True)
    best = compare_models()
    plot_model(best, plot = 'forecast', data_kwargs = {'fh' : 24})
    plot_model(best, plot = 'diagnostics')
    plot_model(best, plot = 'insample')
    final_best = finalize_model(best)
    predict_model(best, fh = 24)
    model_storage_path += "pycaret_model.pkl"
    with open(model_storage_path, 'wb+') as out:
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
    parser.add_argument('-clean_tech_data_store_dir', help='directory to store raw technical data files')
    parser.add_argument('-start_date', help='start date information')
    parser.add_argument('-end_date', help='end_date information')
    parser.add_argument('-model_storage_path', help='directory to store model in pickel format')

    args = parser.parse_args()
    print(args)
    get_best_ts_model(stock_id=args.stock_ids, 
                      clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                        model_storage_path = args.model_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)