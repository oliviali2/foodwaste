import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import seaborn as sns
from sklearn.model_selection import train_test_split
%matplotlib inline

import datetime
from datetime import date, timedelta

dtypes = {'store_nbr': np.dtype('int64'),
          'item_nbr': np.dtype('int64'),
          'unit_sales': np.dtype('float64'),
          'onpromotion': np.dtype('O')}

train = pd.read_csv('data/train.csv', dtype=dtypes)
test = pd.read_csv('data/test.csv', dtype=dtypes)
# stores = pd.read_csv('data/stores.csv')
items = pd.read_csv('dataitems.csv')
# trans = pd.read_csv('data/transactions.csv')
#oil = pd.read_csv('../input/oil.csv') #we upload this database later
holidays = pd.read_csv('data/holidays_events.csv')

#Using a subset of the data
date_mask = (train['date'] >= '2017-07-15') & (train['date'] <= '2017-08-15')
pd_train = train[date_mask]

#Merge train
pd_train = pd_train.drop('id', axis = 1)
pd_train = pd_train.merge(stores, left_on='store_nbr', right_on='store_nbr', how='left')
pd_train = pd_train.merge(items, left_on='item_nbr', right_on='item_nbr', how='left')
pd_train = pd_train.merge(holidays, left_on='date', right_on='date', how='left')
pd_train = pd_train.merge(oil, left_on='date', right_on='date', how='left')
pd_train = pd_train.drop(['description', 'state', 'locale_name', 'class'], axis = 1)
