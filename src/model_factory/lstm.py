import tqdm
import pandas as pd
import numpy as np
from numpy import log
import io
import pickle

import yfinance as yf
import requests

import matplotlib.pyplot as plt
from matplotlib import pyplot
#import altair as alt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler
from math import sqrt

import torch
import torch.nn as nn
from torch.autograd import Variable

from  feature_n_test_train_provider import return_train_test_with_feature


def get_lstm_model(stock_id =['TSLA'], 
               start_date ='2008-03-01', 
               end_date ='2023-06-20', 
               clean_tech_data_store_dir='data/clean_data',
               model_storage_path = 'models/'):
    for id in stock_id:
        clean_file_path = clean_tech_data_store_dir + "/tech_fundamental_sentiment_" + id + "_"+start_date +"_" +end_date
    print(clean_file_path)
    df = pd.read_csv(clean_file_path)
    data = df.filter(['Close','score'])
    data.Close = data.Close.pct_change(periods=6)
    # get three day forecast not just for tomorrow
    data.Close = data.Close.shift(-6) 
    data = data.dropna()

    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .9 ))
    #X_train = dataset[:int(np.ceil( len(dataset) * .9 ))]
    #y_test = dataset[int(np.ceil( len(dataset) * .9 )):]
# function to create train, test data given stock data and sequence length
    look_back = 40 # choose sequence length
    x_train, y_train, x_test, y_test = load_data(dataset, look_back)
    print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    y_train.size(),x_train.size()
    # Build model#####################
    input_dim = 2
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())
    train_model(look_back, model, x_train,y_train,loss_fn,optimiser)
    model_storage_path += "lstm_model.pkl"
    with open(model_storage_path, 'wb+') as out:
        pickle.dump(model, out)
    predict(model,x_test,y_test)

def predict(model,x_test,y_test):
    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    #y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    #y_train = scaler.inverse_transform(y_train.detach().numpy())
    #y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    #y_test = scaler.inverse_transform(y_test.detach().numpy())

    y_test_pred = (y_test_pred.detach().numpy())
    y_test = (y_test.detach().numpy())

    testScore = np.sqrt(mean_squared_error(y_test, y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    print(f"R2 SCORE: {r2_score(y_test, y_test_pred[:,0])}")
    real =[]
    predict =[]
    for i in range(y_test_pred.shape[0]):
        #real.append(x_test[i].item())
        predict.append(y_test_pred[i].item())
            #print(f"{y_pred[i].item()} (expected {y_test[i].numpy()})")
    data_frame = pd.DataFrame(y_test,y_test_pred[:,0])
    data_frame = data_frame.reset_index()
    data_frame.columns =['real','predicted']

    data_frame.head()
    data_frame.plot()
    plt.show()
    data_frame.plot(title = "LSTM Prediction").get_figure().savefig('data/visualization/output.png')


def load_data(stock, look_back):
    data_raw = stock
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(0.1*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,0]

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,0]
    return [x_train, y_train, x_test, y_test]

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions

        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.1)


    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

def train_model(look_back, model, x_train,y_train,loss_fn,optimiser):
    num_epochs = 1000
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    seq_dim =look_back-1

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        #model.hidden = model.init_hidden()

        # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred[:,0], y_train)
        if t % 10 == 0 and t !=0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()
    plt.plot(hist, label="Training loss")
    plt.legend()
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
    get_lstm_model(stock_id=args.stock_ids, 
                      clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                        model_storage_path = args.model_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)