import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import copy
import tqdm
import pickle

from  feature_n_test_train_provider import return_train_test_with_feature
from log_metrics import log_metrics, get_smape

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def get_ann_model(stock_id =['TSLA'], 
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
    df_train, df_test = return_train_test_with_feature(df,stock_tickr)
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
    model = None
    if stock_tickr == 'TSLA':
        model = Regressor(26)
    else:
        model = Regressor(26)

    print(model)
    train_model(model, X, y,stock_tickr)
    predict_result(model,X_test,y_test,stock_tickr,scaler,scaler_y)
    model_storage_path += "ann_model.pt"
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(model_storage_path) # Save

# define the model
def predict_result(model,X_test, y_test,stock_tickr,scaler,scaler_y):
    loss_fn   = nn.MSELoss()
    with torch.no_grad():
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
    data_frame = pd.DataFrame(real,predict)
    print(data_frame.head())
    #metric = R2Score()
    #metric.update(y_pred, y_test)
    #r2 = metric.compute().detach()
    #print(r2)
    rmse = np.sqrt(mean_squared_error(real, predict))
    r2 = r2_score(real, predict)
    print(r2) 
    mape = mean_absolute_percentage_error(real,predict)
    smape = get_smape(real,predict)
    print("r2 =", r2, "===rmse =", rmse, "mape =", mape, "smape =", smape )

    log_metrics(stock_tickr, 'ANN_model_test', rmse, r2, mape, smape)

    data_frame = data_frame.reset_index()
    data_frame.columns =['real Close','predicted Close']
    data_frame.head()
    data_frame.plot() 
    plt.savefig("data/visualization/"+(stock_tickr + '_ANN real vs predicted Close'))

class Regressor(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.hidden1 = nn.Linear(layers, 13)
        self.act1 = nn.Tanh()
        #self.hidden2 = nn.Linear(13, layers)
        #self.act2 = nn.ReLU()
        self.output = nn.Linear(13, 1)
        self.act_output = nn.Tanh()

    def forward(self, x):
        #x = self.dropout(x)
        x = self.act1(self.hidden1(x))
        x = self.dropout(x)
        #x = self.act2(self.hidden2(x))
        #x = self.act3(self.hidden3(x))        
        #x = self.dropout(x)
        x = self.act_output(self.output(x))
        return x


def train_model(model, X, y, stock_tickr):
        
    history = []

    # train the model
    loss_fn   = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 1100
    batch_size = 20
    batch_start = torch.arange(0, len(X), batch_size)

    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X[start:start+batch_size]
                y_batch = y[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X)
        mse = loss_fn(y_pred, y)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
 
    plt.plot(history)
    plt.savefig("data/visualization/"+(stock_tickr + '_ANN_log_ret loss optimization'))

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
    get_ann_model(stock_id=args.stock_ids, 
                      clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                        model_storage_path = args.model_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)