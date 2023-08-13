import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = pd.read_csv("./models_evalution.csv")
#df = df[['Model','R^2','Evaluated On']]
df = df[df['Evaluated On']=='F']
#df = df.drop(columns=['Model'])
#df = df.groupby('Model')
print(df.head())
axes = df.plot.bar(x='Model', rot=0,figsize=(5, 3), \
title="Evaluation result for F",style="fivethirtyeight")  
#df.plot.bar(rot=0, subplots=True)
plt.show()