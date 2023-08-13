import pandas as pd
import numpy as np

metric_file = 'data/model/metrics.csv'

def log_metrics(tickr,model,RMSE,R2,MAPE,SMAPE):
    data = {
    'Tickr': [tickr], 
    'Model': [model],
    'RMSE': [RMSE],
    'R^2': [R2],
    'MAPE': [MAPE],
    'SMAPE':[SMAPE]
    }
    df = pd.DataFrame(data)
    with open(metric_file, 'a') as f:
        f.write('\n')    
    df.to_csv(metric_file, mode='a', index=False, header=False)
#log_metrics('lstm',2,1)

def get_smape(actual,forecast):
    return 100/len(actual) * np.sum(2 * np.abs(np.array(forecast) - np.array(actual)) / \
                                    (np.abs(actual) + np.abs(forecast)))
