'''
This code works to fill in NaN and empty values.
It is kept separate from clean_train_medium.py for convenience of execution.
To clean the actual train dataset, open and run clean_train.py
'''


import pandas as pd
import numpy as np

train = pd.read_csv('data/cleaned_train_medium.csv')

train['family'] = train['family'].replace(np.nan, 'GROCERY I', regex=True)
#type_holiday column: fill missing values w/ 'Regular'
train['type_holiday'] = train['type_holiday'].replace(np.nan, 'Regular', regex=True)
train['locale'] = train['locale'].replace(np.nan, 'None', regex=True)
train['perishable'] = train['perishable'].fillna(0)
# train = train.drop('transferred',axis=1)

has_n = train.columns[train.isna().any()].tolist()
print('these cols have nan:', has_n)


'''Write results to cleaned_train_medium.csv '''
train.to_csv('data/cleaned_train_medium.csv', index=False)
print('****************************')
