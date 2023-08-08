import json
import math
import os
import pickle
import sys

import pandas as pd
from sklearn import metrics
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt
import numpy as np
        
def forecast_one_step(model):
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])

def evaluate(stock_id =['TSLA'], 
               start_date ='2018-06-20', 
               end_date ='2023-08-05', 
               clean_tech_data_store_dir='data/clean_data/all_combined',
               model_file = 'models/time_series_model.pkl',
               live ='live_path',
               save_path='save_path'):
    for id in stock_id:
        clean_file_path = clean_tech_data_store_dir + "/tech_fundamental_sentiment_" + id + "_"+start_date +"_" +end_date
        df = pd.read_csv(clean_file_path)
        X_train = df[:int(len(df)*.9)]['Close']
        labels = df[int(len(df)*.9):]['Close']
        print(X_train.head())
        print(labels.head())
    # Load model and data.
    with open(model_file, "rb") as fd:
        model = pickle.load(fd)
    predictions = []
    confidence_intervals = []

    for new_ob in labels:
        fc, conf = forecast_one_step(model)
        predictions.append(fc)
        confidence_intervals.append(conf)

        # Updates the existing model with real data after related prediction
        model.update(new_ob)

    # Use dvclive to log a few simple metrics...
    mse = metrics.mean_squared_error(labels, predictions)
    r2_score = metrics.r2_score(labels, predictions)
    if not live.summary:
        live.summary = {"MSE": {}, "r2_score": {}}
    live.summary["MSE"] = mse
    live.summary["r2_score"] = r2_score



def save_importance_plot(model, save_path):
    """
    Save feature importance plot.
    """
    fig, axes = plt.subplots(dpi=100)
    model.plot_diagnostics(figsize=(10,8))
    fig.savefig(os.path.join(save_path, "importance.png"))

def Eval1():
    EVAL_PATH = "eval"

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model save_path\n")
        sys.exit(1)

    model_file = sys.argv[1]
    # Load model and data.
    with open(model_file, "rb") as fd:
        model = pickle.load(fd)

#    with open(train_file, "rb") as fd:
#        train, feature_names = pickle.load(fd)

#    with open(test_file, "rb") as fd:
#        test, _ = pickle.load(fd)

    # Evaluate train and test datasets.
    live = Live(os.path.join(EVAL_PATH, "live"), dvcyaml=False)
    evaluate (stock_id =['TSLA'], 
               start_date ='2018-06-20', 
               end_date ='2023-08-05', 
               clean_tech_data_store_dir='data/clean_data/all_combined',
               model_file = 'models/time_series_model.pkl',
               live=live,
               save_path=eval)
    live.make_summary()

    # Dump feature importance plot.
    save_importance_plot(model, save_path=EVAL_PATH)


if __name__ == "__main__":
    Eval1()