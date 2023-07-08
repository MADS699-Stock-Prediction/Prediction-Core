import pmdarima as pm
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
from sklearn.metrics import r2_score
import json
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import pyplot

def train_auto_arima(stock_id =['TSLA'], 
               start_date ='2008-03-01', 
               end_date ='2023-06-20', 
               clean_tech_data_store_dir='data/clean_data',
               model_storage_path = 'models/'):
    for id in stock_id:

        clean_file_path = clean_tech_data_store_dir + "/tech_indicator_" + id + "_"+start_date +"_" +end_date
        df = pd.read_csv(clean_file_path)
        X_train = df[:int(len(df)*.9)]
        y_test = df[int(len(df)*.9):]
        print(X_train.head())
        print(y_test.head())
        model = pm.auto_arima(X_train['Close'], start_p=1, start_q=1,
                        exogenous= X_train['Volume'],
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=4, max_q=4, # maximum p and q
                        # m=80,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)

        print(model.summary())
        model_output_path = "models/time_series_model.pkl"
        with open(model_output_path, 'wb+') as out:
            pickle.dump(model, out)
        return model, y_test

def plot_info(model):
    model.plot_diagnostics(figsize=(10,8))
    plt.show()

def forecast_one_step(model):
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])

def get_stats(model,y_test):
    forecasts = []
    confidence_intervals = []

    for new_ob in y_test['Close']:
        fc, conf = forecast_one_step(model)
        forecasts.append(fc)
        confidence_intervals.append(conf)

        # Updates the existing model with real data after related prediction
        model.update(new_ob)

    print(f"Mean squared error: {mean_squared_error(y_test['Close'], forecasts)}")
    print(f"SMAPE: {smape(y_test['Close'], forecasts)}")
    print(f"R2 SCORE: {r2_score(y_test['Close'], forecasts)}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock_ids",
        '--names-list',
        nargs="*",  
        default=['TSLA'],
        )
    parser.add_argument('-model_storage_path', help='directory to store model in pickel format')
    parser.add_argument('-clean_tech_data_store_dir', help='directory to store clean technical data files')
    parser.add_argument('-start_date', help='start date information')
    parser.add_argument('-end_date', help='end_date information')
    args = parser.parse_args()    
    print(args)
    model, y_test = train_auto_arima(stock_id=args.stock_ids, 
                    clean_tech_data_store_dir= args.clean_tech_data_store_dir, 
                    model_storage_path = args.model_storage_path, 
                    start_date = args.start_date, 
                    end_date = args.end_date)
    plot_info(model)
    get_stats(model,y_test)
