
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import copy
import tqdm
import pickle
from  feature_n_test_train_provider import return_train_test_with_feature_for_evaluate
from log_metrics import log_metrics, get_smape

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

def evaluate_ann_model(stock_id =['TSLA'], 
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
    df_train = df_train.tail(200)
    y = df_train[['log_ret']]
    X = df_train.drop(['log_ret'], axis=1)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X = scaler.fit_transform(X)
    print(X.shape)
    X = torch.Tensor(X)
    y = scaler_y.fit_transform(y)
    y = torch.tensor(y, dtype=torch.float32)#.reshape(-1, 1).clone().detach()

    y_test = df_test[['log_ret']]
    X_test = df_test.drop(['log_ret'],axis=1)

    print(type(y_test))
    X_test = scaler.transform(X_test)
    X_test = torch.Tensor(X_test)
    y_test = scaler_y.transform(y_test)

    y_test = torch.tensor(y_test, dtype=torch.float32)#.reshape(-1,1)

    model_file = model_storage_path + "ann_model.pt"
    print(model_file) 
    model = torch.jit.load(model_file)

    predict_result(model,X_test,y_test,stock_tickr,scaler,scaler_y)

def predict_result(model,X_test, y_test,stock_tickr,scaler,scaler_y):
    loss_fn   = nn.MSELoss()
    with torch.no_grad():
        # Test out inference with 5 samples
        y_pred = model(X_test)
        mse= float(loss_fn(y_pred, y_test))
        print("MSE: %.2f" % mse)
        print("RMSE: %.2f" % np.sqrt(mse))
        y_pred = scaler_y.inverse_transform(y_pred)
        y_test = scaler_y.inverse_transform(y_test)

        real =[]
        predict =[]
        for i in range(y_test.shape[0]):
            #print(i)
            real.append(y_test[i].item())
            predict.append(y_pred[i].item())
            #print(f"{y_pred[i].item()} (expected {y_test[i].numpy()})")
    data_frame = pd.DataFrame(real,predict)
    data_frame = data_frame.reset_index()
    data_frame.columns =['real Close','predicted Close']
    data_frame.head()
    data_frame.plot()
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
 
    plt.savefig("data/visualization/"+ str(current_time) + \
                 (stock_tickr + '_ANN evaluation real vs predicted Close'))
    rmse = np.sqrt(mean_squared_error(real, predict))
    r2 = r2_score(real, predict)
    mape = mean_absolute_percentage_error(real,predict)
    smape = get_smape(real,predict)
    print("r2 =", r2, "===rmse =", rmse, "mape =", mape, "smape =", smape )

    log_metrics(stock_tickr, 'ANN_model_evaluate', rmse, r2, mape, smape)

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
    evaluate_ann_model(stock_id=args.stock_ids, 
                      clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                        model_storage_path = args.model_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)