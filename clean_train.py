'''
This code works to clean the train_small and train_medium datasets.
To clean the actual train dataset, open and run clean_train.py
'''

import pandas as pd
import numpy as np

# dtypes = {'store_nbr': np.dtype('int64'),
#           'item_nbr': np.dtype('int64'),
#           'unit_sales': np.dtype('float64'),
#           }


pd_train = pd.read_csv('data/train.csv')
pd_train = pd_train['onpromotion'].fillna(-1)

stores = pd.read_csv('data/stores.csv')
items = pd.read_csv('data/items.csv')
# trans = pd.read_csv('data/transactions.csv')
oil = pd.read_csv('data/oil.csv')
holidays = pd.read_csv('data/holidays_events.csv')


print('columns before merging')
print(list(pd_train.columns.values))

pd_train['store_nbr'] = pd_train['store_nbr'].astype(np.uint8)
# The ID column is a continuous number from 1 to 128867502 in train and 128867503 to 125497040 in test
# pd_train['id'] = pd_train['id'].astype(np.uint32)
# item number is unsigned
pd_train['item_nbr'] = pd_train['item_nbr'].astype(np.uint32)
#Converting the date column to date format
# pd_train['date']=pd.to_datetime(pd_train['date'],format="%Y-%m-%d")



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
pd_train.to_csv('cleaned_train.csv')

print('after merging ')
print(list(pd_train.columns.values))
print('# of lines in cleaned dataframe: '+ str(pd_train.count()))
print(pd_train.values[0])
