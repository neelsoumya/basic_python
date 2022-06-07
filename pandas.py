# In pandas, split into test and training using pandas.dataframe.sample

import pandas as pd

data =  pd.read_csv('/Users/soumya/Documents/abalone.data', header = None)

# split into test and training

data.sample(frac=0.7)
