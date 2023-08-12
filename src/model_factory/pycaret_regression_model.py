# can not install on the local system works on colab only
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from pycaret.regression import *
import pandas as pd
import pickle
import IPython
import matplotlib_inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib

from log_metrics import log_metrics, get_smape

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

scaler = MinMaxScaler()

from  feature_n_test_train_provider import return_train_test_with_feature

# need to remove hardcoding


def get_best_regression_model(stock_id =['TSLA'], 
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
    df_new = df[['Close','score']]
    df_new = df_new.tail(50)
    #print(df_new.tail(30))
    df_new= scaler.fit_transform(df_new)
    df_new1 = pd.DataFrame(df_new, columns = ['Close','score'])

    df_new1.plot(title ="MinMax(0,1) Close vs Sentiment Score ")
    plt.savefig('MinMax(0,1) Close vs Sentiment Score.png')

    df_train, df_test = return_train_test_with_feature(df,stock_tickr)
    s = setup(df_train, target="log_ret", session_id=1234,fold_strategy='timeseries', \
              normalize=True,log_experiment=True,experiment_name="PyCaretWithSentiment", feature_selection=True)
    best = compare_models()
    #dashboard(best)
    evaluate_model(best)
    model_storage_path += "pycaret_regression_model.pkl"
    with open(model_storage_path, 'wb+') as out:
        pickle.dump(best, out)
    save_model(best, "pycaret_regression_model.pkl")
    #model = load_model("pycaret_regression_model.pkl")

    pred_hostdout = predict_model(best,data=df_train)
    print(df_test.head())

    real_test_y = df_test['log_ret']
    real = real_test_y            
    new_data= df_test.drop(['log_ret'], axis=1)
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    predictions = predict_model(best, data = new_data)
    predict = predictions['prediction_label']
    df_final_predict=pd.concat([real_test_y,predict],axis=1)
    print(df_final_predict.head())

    df_final_predict.columns=['Close','Predicted_close']
    plot = df_final_predict.plot(title= (str(current_time) + "_" + \
                                         stock_tickr+ "pyCaret regression model test")).get_figure()
    
    plot.savefig("data/visualization/"+ str(current_time) + "_" + stock_tickr + "_ pyCaret regression model test")

    rmse = np.sqrt(mean_squared_error(real, predict))
    r2 = r2_score(real, predict)
    mape = mean_absolute_percentage_error(real,predict)
    smape = get_smape(real,predict)
    print("r2 =", r2, "===rmse =", rmse, "mape =", mape, "smape =", smape )

    log_metrics(stock_tickr, 'pycaret_regression_model_test', rmse, r2, mape, smape)

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
    get_best_regression_model(stock_id=args.stock_ids, 
                      clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                        model_storage_path = args.model_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)