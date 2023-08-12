
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pycaret.regression import *

from  feature_n_test_train_provider import return_train_test_with_feature_for_evaluate
from log_metrics import log_metrics, get_smape

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

def evaluate_pycaret_regression_model(stock_id =['TSLA'], 
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

    df_train, df_test = return_train_test_with_feature_for_evaluate(df,stock_tickr)
    model = load_model("pycaret_regression_model.pkl")
    print(df_test.head())

    real = df_test['log_ret']
    new_data= df_test.drop(['log_ret'], axis=1)
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    predictions = predict_model(model, data = new_data)
    predict = predictions['prediction_label']
    df_final_predict=pd.concat([real,predict],axis=1)
        
    df_final_predict.columns=['Close','Predicted_close']
    plot = df_final_predict.plot(title= (str(current_time) + "_" + \
                                         stock_tickr+ "pyCaret regression model evaluation")).get_figure()
    
    plt.savefig("data/visualization/"+ str(current_time) + "_" + stock_tickr + "_ pyCaret regression model evaluation")
    rmse = np.sqrt(mean_squared_error(real, predict))
    r2 = r2_score(real, predict)
    mape = mean_absolute_percentage_error(real,predict)
    smape = get_smape(real,predict)
    print("r2 =", r2, "===rmse =", rmse, "mape =", mape, "smape =", smape )

    log_metrics(stock_tickr, 'pycaret_regression_evaluate', rmse, r2, mape, smape)


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
    evaluate_pycaret_regression_model(stock_id=args.stock_ids, 
                      clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                        model_storage_path = args.model_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)