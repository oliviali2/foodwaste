'''
This code works to fill in NaN and empty values.
It is kept separate from clean_train_medium.py for convenience of execution.
To clean the actual train dataset, open and run clean_train.py
'''


import pandas as pd
import numpy as np

train = pd.read_csv('data/cleaned_train_medium.csv')

train['family'] = train['family'].fillna(train.family.mode())
#type_holiday column: fill missing values w/ 'Regular'
train['type_holiday'] = train['type_holiday'].replace(np.nan, 'Regular', regex=True)
train['locale'] = train['locale'].replace(np.nan, 'NA', regex=True)

train = train.drop('transferred',axis=1)

'''Write results to cleaned_train_medium.csv '''
train.to_csv('data/cleaned_train_medium.csv', index=False)
print('****************************')
