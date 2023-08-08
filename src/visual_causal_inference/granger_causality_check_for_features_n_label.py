import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

def perform_granger_causality(stock_id =['TSLA'], 
               start_date ='2018-06-20', 
               end_date ='2023-08-05', 
               clean_tech_data_store_dir='../data/clean_data/all_combined',
               causality_storage_path = '../data/clean_data/causal_inference/'):
    for id in stock_id:
        clean_file_path = clean_tech_data_store_dir + "/tech_fundamental_sentiment_" + id + "_"+start_date +"_" +end_date
    print(clean_file_path)
    df = pd.read_csv(clean_file_path)
    # as of now only sentiment score and log_ret
    cols_to_drop = ['Date','Date.1','Close','High','Low','Open','Adj Close','totalasset','shares','dps','eps','ebitda','bv']
    #variables = list(set(df.columns)-set(cols_to_drop))
    #print(variables)
    df = df.tail(100)
    df['log_ret'] = np.log(df.Close) - np.log(df.Close.shift(1))
    calculate_correlations(df,causality_storage_path)
    df = df.drop(columns=cols_to_drop)
    df = df.dropna()
    variables =  df.columns
    test = 'ssr_chi2test'
    #Check with lag 10
    causality_df = check_causality(df,variables,test,True,10) 
    causality_df.to_csv(causality_storage_path + 'granger_causality_output.csv',header=True)

def calculate_correlations(df,causality_storage_path):
    #df.corr().to_csv(causality_storage_path, header=True)
    corr_methods= ['pearson', 'kendall', 'spearman']
    for corr_method in corr_methods:
        print(df.head())
        corr= df.reset_index().drop(columns=['index','Date.1','Date']).corr()
        corr.to_csv(causality_storage_path + corr_method+'_correlations.csv', header=True)
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