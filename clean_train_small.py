'''
This code works to clean the train_small and train_medium datasets.
To clean the actual train dataset, open and run clean_train.py
'''

import pandas as pd
import numpy as np

dtypes = {'store_nbr': np.dtype('int64'),
          'item_nbr': np.dtype('int64'),
          'unit_sales': np.dtype('float64'),
          }

pd_train = pd.read_csv('data/train_small.csv', dtype=dtypes)

# pd_train = pd_train.where((pd.notnull(pd_train)), None)

stores = pd.read_csv('data/stores.csv')
items = pd.read_csv('data/items.csv')
#Not downloaded
trans = pd.read_csv('data/transactions.csv')
oil = pd.read_csv('data/oil.csv')
holidays = pd.read_csv('data/holidays_events.csv')



print('DATATYPES: train')
print(pd_train.dtypes)
print(pd_train.head())
# print('DATATYPES: items')
# print(items.dtypes)

pd_train = pd_train.drop('id', axis = 1)
pd_train = pd_train.merge(stores, left_on='store_nbr', right_on='store_nbr', how='left')
pd_train = pd_train.merge(items, left_on='item_nbr', right_on='item_nbr', how='left')
pd_train = pd_train.merge(holidays, left_on='date', right_on='date', how='left')
pd_train = pd_train.merge(oil, left_on='date', right_on='date', how='left')
pd_train = pd_train.drop(['description', 'state', 'locale_name', 'class', 'dcoilwtico'], axis = 1)

for idx, date in enumerate(pd_train['date']):
  sdate = int(str(date).split("/", 1)[0])
  holidays['date'][idx] = sdate

cols = pd_train.columns.tolist()
cols.insert(len(cols)-1, cols.pop(cols.index('unit_sales')))
pd_train = pd_train[cols]
pd_train = pd_train.where((pd.notnull(pd_train)), None)
pd_train = pd_train.rename(index=str, columns={"type_x": "type_store", "type_y": "type_holiday"})
pd_train.to_csv('data/cleaned_train_small.csv')

print('after merging ')
print(list(pd_train.columns.values))
print('# of lines in cleaned dataframe: '+ str(pd_train.count()))
print('final column names: \n')
print(pd_train.columns.values)
