import numpy as np
import pandas as pd
from numpy import array

import matplotlib.pyplot as plt
from matplotlib import pyplot

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM, Dropout, BatchNormalization, Dense
from keras import optimizers
from keras.layers import Bidirectional

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

from log_metrics import log_metrics, get_smape

from prepare_step_data import prepare_step_data_return 

def train_n_save_model(stock_id =['TSLA'], 
               start_date ='2008-03-01', 
               end_date ='2023-06-20', 
               clean_tech_data_store_dir='../data/clean_data/all_combined',
               model_storage_path = 'models/'):
    stock_tickr = None
    for id in stock_id:
        stock_tickr = id
        clean_file_path = clean_tech_data_store_dir + \
            "/tech_fundamental_sentiment_" + id + "_"+start_date + \
                "_" +end_date
    print(clean_file_path)
    df = pd.read_csv(clean_file_path)
    df = df[['Close']]
    print(df.head())
    scaler = MinMaxScaler()
    df['scaled_close'] = scaler.fit_transform(np.expand_dims(df['Close'].values, axis=1))
    print(df.head())
    #return
    #df['log_ret'] = df['Close'] # np.log(df.Close) - np.log(df.Close.shift(1))
    #df['log_ret'] = df['log_ret'].shift(-3)

    #df['log_ret'] = df['Close']#.pct_change(periods=6)
    df = df.dropna()
    # keep n_steps as 20 assuming 20 days impact is possible for the prediction
    n_steps = 7
    X, y = prepare_step_data_return(df.scaled_close, n_steps)

    train_size = int(np.round(X.shape[0]*(0.8)))
    test_size = X.shape[0] - train_size
    print(X.shape[0], train_size,test_size)

    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]    
    y_test = y[train_size:]
    print(X_train[:5],y_train[:5])

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1 # as of now only one feature i.e. log_ret
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

    # define model
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

    model = Sequential()
    model.add(LSTM(120, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.3))
    model.add(LSTM(240,return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(30))
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mse')
    # fit model
    #model.fit(X_train, y_train, epochs=10,batch_size=10, verbose=1)
    history = model.fit(X_train, y_train, epochs=400, batch_size=10, callbacks=[callback],verbose=1)

    # demonstrate prediction
    yhat = model.predict(X_test, verbose=0)
    
    # calcaulate and update metrics

    #model.save('my_model.keras')
    #new_model = tf.keras.models.load_model('my_model.keras')
    y_test = np.squeeze(scaler.inverse_transform(y_test.reshape(-1,1)))
    yhat = np.squeeze(scaler.inverse_transform(yhat))

    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    r2 = r2_score(y_test,yhat)
    mape = mean_absolute_percentage_error(y_test,yhat)
    smape = get_smape(y_test,yhat)
    print("r2 =", r2, "===rmse =", rmse, "mape =", mape, "smape =", smape )

    log_metrics(stock_tickr, 'LSTM_Univariate_model_test', rmse, r2, mape, smape)


    data_frame = pd.DataFrame(y_test,yhat)
    data_frame = data_frame.reset_index()
    data_frame.columns =['Real Close','Predicted Close']

    plot =data_frame.plot(title = (stock_tickr + "_LSTM_univariate_model_test_data_prediction")).get_figure()
    save_path = 'data/visualization/' + stock_tickr + "_Vanilla_LSTM(univariate_Close_7day_lookbck) real vs predicted"
    plot.savefig(save_path)  
    model_storage_path = model_storage_path + "trained_on" + \
        "_lstm_univariate_vanilla.pt"
    model.save(model_storage_path)

    #reloaded_artifact = tf.saved_model.load("exported_model")

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
    train_n_save_model(stock_id=args.stock_ids, 
                      clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                        model_storage_path = args.model_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)