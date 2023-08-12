# can not install on the local system works on colab only
from pycaret.time_series import *
from pycaret import *
import pandas as pd
import pickle
import numpy as np
from log_metrics import log_metrics, get_smape

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error


def get_best_ts_model(stock_id =['TSLA'], 
               start_date ='2008-03-01', 
               end_date ='2023-06-20', 
               clean_tech_data_store_dir='data/clean_data',
               model_storage_path = 'models/'):
    stock_tickr = None
    for id in stock_id:
        stock_tickr = id
        clean_file_path = clean_tech_data_store_dir + "/tech_fundamental_sentiment_" + id + "_"+start_date +"_" +end_date
    print(clean_file_path)
    df = pd.read_csv(clean_file_path)
    df = df[['Close']]
    print(df.head())
    #df = df.asfreq('B')
    print(df.head())
    df = df[['Close']]
    print(df.head())
    real = df.tail(6)
    #print(real)
    #return
    df = df[:(df.shape[0]-6)]
    df['log_ret'] = df.Close #np.log(df.Close) - np.log(df.Close.shift(1))

    s = setup(data =df.log_ret , fh = 3, fold = 5, session_id = 123, \
              experiment_name="pycaret_timeseries",numeric_imputation_target ='ffill', log_experiment=True)
    best = compare_models()
    plot_model(best, plot = 'forecast', data_kwargs = {'fh' : 24},save=True,return_fig=True)
    plot_model(best, plot = 'diagnostics',save=True,return_fig=True)
    plot_model(best, plot = 'insample',save=True,return_fig=True)
    final_best = finalize_model(best)
    predict =(predict_model(final_best, fh = 6,return_pred_int=True)['y_pred'])

    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    df_final_predict=pd.concat([real,predict],axis=1)
    print(df_final_predict.head())

    df_final_predict.columns=['Close','Predicted_close']
    plot = df_final_predict.plot(title= (str(current_time) + "_" + \
                                         stock_tickr+ "pyCaret timeseries model test")).get_figure()
    
    plot.savefig("data/visualization/"+ str(current_time) +"_" + stock_tickr + "_ pyCaret timeseries model test")

    rmse = np.sqrt(mean_squared_error(real, predict))
    r2 = r2_score(real, predict)
    mape = mean_absolute_percentage_error(real,predict)
    #smape = get_smape(real,predict)
    print("r2 =", r2, "===rmse =", rmse, "mape =", mape, "smape =", "NA" )

    log_metrics(stock_tickr, 'pycaret_ts_model_test', rmse, r2, mape, "NA")

    model_storage_path += "pycaret_model.pkl"
    save_model(best, "pycaret_ts_model.pkl")
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