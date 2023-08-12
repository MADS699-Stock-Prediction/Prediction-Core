import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def perform_granger_causality(stock_id =['TSLA'], 
               start_date ='2018-06-20', 
               end_date ='2023-08-05', 
               clean_tech_data_store_dir='../data/clean_data/all_combined',
               causality_storage_path = '../data/clean_data/causal_inference/'):
    stock_tickr = None
    for id in stock_id:
        clean_file_path = clean_tech_data_store_dir + "/tech_fundamental_sentiment_" + id + "_"+start_date +"_" +end_date
        stock_tickr = id
    print(clean_file_path)
    df = pd.read_csv(clean_file_path)
    # as of now only sentiment score and log_ret
    cols_to_drop = ['Unnamed: 0','Date','Date.1','Close','High','Low','Open','Adj Close']
    if stock_tickr =='TSLA':
        cols_to_drop = ['Unnamed: 0','Date','Date.1','Close','High','Low','Open','Adj Close','totalasset','shares','dps','eps','ebitda','bv']
    #variables = list(set(df.columns)-set(cols_to_drop))
    #print(variables)
    df = df.tail(365)
    df['log_ret'] = np.log(df.Close) - np.log(df.Close.shift(1))
    plot_sentiment_vs_log_ret(df,causality_storage_path,stock_tickr)
    calculate_correlations(df,causality_storage_path,stock_tickr)

    df = df.drop(columns=cols_to_drop)
    df = df.dropna()
    print(df.columns)
    df = df.loc[:, (df != df.iloc[0]).any()] 
    print(df.columns)
    variables =  df.columns
    lag_period = 10
    test = 'ssr_chi2test'
    causality_df = check_causality(df,variables,test,True,lag_period) 
    causality_df.to_csv(causality_storage_path + stock_tickr + str(lag_period) + \
                 'granger_causality_output.csv',header=True)

def plot_sentiment_vs_log_ret(df,causality_storage_path,stock_tickr):
    log_ret_n_setiment = ['log_ret', 'score']
    df_for_graph = df[log_ret_n_setiment] 
    scaler = MinMaxScaler(feature_range=(-1,1))
    df_for_graph = scaler.fit_transform(df_for_graph)
    df_for_graph =pd.DataFrame(df_for_graph,columns=log_ret_n_setiment)

    plot = df_for_graph.plot(title = stock_tickr + "_log_ret and sentiment score")   
    fig = plot.get_figure()
    fig.savefig(causality_storage_path + stock_tickr + "_log_ret and sentiment score.png")
    plt.show()

def calculate_correlations(df,causality_storage_path,tickr):
    #df.corr().to_csv(causality_storage_path, header=True)
    corr_methods= ['pearson', 'kendall', 'spearman']
    for corr_method in corr_methods:
        corr= df.reset_index().drop(columns=['index','Date.1','Date']).corr()
        corr.to_csv(causality_storage_path +tickr + '_' + corr_method + \
                    '_correlations.csv', header=True)

def check_causality(main_df,variables,test,log,maxlag):
    """
        Check Granger Causality of all possible combinations of the Time series
        based on passed variables.
        The rows are the response variable, columns are predictors. The values in the 
        table the P-Values. P-Values lesser than the significance level (0.05), implies
        the Null Hypothesis that the coefficients of the corresponding past values is
        zero, that is, the X does not cause Y can be rejected.

        main_df        : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
        test      : the test to perform
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            print('Working for ==>',r)
            test_result = grangercausalitytests(main_df[[r, c]], maxlag=maxlag)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            #if log: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock_ids",
        '--names-list',
        nargs="*",  
        default=['TSLA'],
        )
    parser.add_argument('-clean_tech_data_store_dir', help='directory to read combined \
                         processed data file')
    parser.add_argument('-start_date', help='start date information')
    parser.add_argument('-end_date', help='end_date information')
    parser.add_argument('-causality_storage_path', help='directory to store causality in \
                        csv format with header')

    args = parser.parse_args()
    print(args)
    
    perform_granger_causality(stock_id=args.stock_ids, 
                      clean_tech_data_store_dir= args.clean_tech_data_store_dir,
                        causality_storage_path = args.causality_storage_path, 
                        start_date = args.start_date,
                        end_date = args.end_date)