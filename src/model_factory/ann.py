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

def get_ann_model(stock_id =['TSLA'], 
               start_date ='2008-03-01', 
               end_date ='2023-06-20', 
               clean_tech_data_store_dir='data/clean_data',
               model_storage_path = 'models/'):
    for id in stock_id:
        clean_file_path = clean_tech_data_store_dir + "/tech_fundamental_sentiment_" + id + "_"+start_date +"_" +end_date
    print(clean_file_path)
    df = pd.read_csv(clean_file_path)
    df_train, df_test = return_train_test_with_feature(df)
    y = df_train['CUMPCTRET_6']
    y = torch.Tensor(y.to_numpy())
    print(type(y))
    X = df_train.drop(['CUMPCTRET_6'], axis=1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    print(X.shape)
    X = torch.Tensor(X)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)#.clone().detach()
    y_test = df_test['CUMPCTRET_6']
    X_test = df_test.drop(['CUMPCTRET_6'],axis=1)

    print(type(y_test))
    X_test = scaler.fit_transform(X_test)
    X_test = torch.Tensor(X_test)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1,1)
    model = Regressor()
    print(model)
    train_model(model, X, y)
    predict_result(model,X_test,y_test)
    model_storage_path += "ann_model.pkl"
    with open(model_storage_path, 'wb+') as out:
        pickle.dump(model, out)

# define the model
def predict_result(model,X_test, y_test):
    loss_fn   = nn.MSELoss()
    with torch.no_grad():
        # Test out inference with 5 samples
        y_pred = model(X_test)
        #print(y_pred)
        #print(y_pred)
        #print(y_test)
        mse= float(loss_fn(y_pred, y_test))
        print("MSE: %.2f" % mse)
        print("RMSE: %.2f" % np.sqrt(mse))
        real =[]
        predict =[]
        for i in range(y_test.shape[0]):
            #print(i)
            real.append(y_test[i].item())
            predict.append(y_pred[i].item())
            #print(f"{y_pred[i].item()} (expected {y_test[i].numpy()})")
    data_frame = pd.DataFrame(real,predict)
    data_frame = data_frame.reset_index()
    data_frame.columns =['real','predicted']
    data_frame.head()
    data_frame.plot() 
    plt.show()   
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.hidden1 = nn.Linear(31, 62)
        self.act1 = nn.Tanh()
        self.output = nn.Linear(62, 1)
        self.act_output = nn.Tanh()

        """
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(31,15 )
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(15, 8)
        self.act4 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.ReLU()
"""
    def forward(self, x):
        #x = self.dropout(x)
        x = self.act1(self.hidden1(x))
        """
        x = self.act2(self.hidden2(x))
        x = self.dropout(x)

        x = self.act3(self.hidden3(x))
        x = self.dropout(x)

        x = self.act4(self.hidden4(x))
        x = self.dropout(x)
"""
        x = self.act_output(self.output(x))
        return x


def train_model(model, X, y):
        
    history = []

    # train the model
    loss_fn   = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 1000
    batch_size = 10
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
    plt.show()

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