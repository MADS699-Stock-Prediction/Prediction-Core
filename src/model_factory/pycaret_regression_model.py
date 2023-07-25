# can not install on the local system works on colab only
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from pycaret.regression import *
import pandas as pd
import pickle
import IPython
import matplotlib_inline
import matplotlib.pyplot as plt

from  feature_n_test_train_provider import return_train_test_with_feature

# need to remove hardcoding


def get_best_regression_model(stock_id =['TSLA'], 
               start_date ='2008-03-01', 
               end_date ='2023-06-20', 
               clean_tech_data_store_dir='data/clean_data',
               model_storage_path = 'models/'):
    for id in stock_id:
        clean_file_path = clean_tech_data_store_dir + "/tech_indicator_" + id + "_"+start_date +"_" +end_date
    print(clean_file_path)
    df = pd.read_csv(clean_file_path)
    df_train, df_test = return_train_test_with_feature(df)
    s = setup(df_train, target="CUMPCTRET_6", session_id=1234,fold_strategy='timeseries', normalize=True,log_experiment=True)
    best = compare_models()
    #dashboard(best)
    evaluate_model(best)
    model_storage_path += "pycaret_regression_model.pkl"
    with open(model_storage_path, 'wb+') as out:
        pickle.dump(best, out)

    pred_hostdout = predict_model(best,data=df_train)

    real_test_y = df_test['CUMPCTRET_6']            
    new_data= df_test.drop(['CUMPCTRET_6'], axis=1)

    #plot_model(best)
    predictions = predict_model(best, data = new_data)
    #plot_model(best)
    df_final_predict=pd.concat([real_test_y,predictions],axis=1)

    #df_final_predict.columns= ['real','predicted']
    df_final_predict[['CUMPCTRET_6','prediction_label']].plot()

    df_pre = df_final_predict[['prediction_label']]
    df_pre['real'] = real_test_y
    df_pre = df_pre.dropna()
    df_pre.head()
    df_pre[:10].plot()
    print("r^2",r2_score(df_pre.real, df_pre.prediction_label), "mse",mean_squared_error(df_pre.real, df_pre.prediction_label, squared=False))
    df_pre.plot()
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
    get_best_regression_model(stock_id=args.stock_ids, 
                      clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                        model_storage_path = args.model_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)