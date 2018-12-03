#support vector regression
#should we use linear SVR?

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn import preprocessing


dataset = pd.read_csv('data/cleaned_train_medium.csv')
has_n = dataset.columns[dataset.isna().any()].tolist()
print('these cols have nan:', has_n)

enc = OneHotEncoder(handle_unknown = 'ignore')

X = dataset.iloc[:, 1:11].values
y = dataset.iloc[:, 11].values




enc.fit(X)
#print enc.categories_
onehotlabels = enc.transform(X).toarray()
onehotlabels.shape

X = onehotlabels

scaler = preprocessing.StandardScaler().fit(X) # Don't cheat - fit only on training data
X = scaler.transform(X) # apply same transformation to test data

numFolds = 2

#kfold splits your data into train and develop for you
kf = KFold(n_splits = numFolds)
kf.get_n_splits(X)
for train_index, dev_index in kf.split(X):
	X_train, X_dev = X[train_index], X[dev_index]
	y_train, y_dev = y[train_index], y[dev_index]

#clf = SVR(gamma = 'scale', C = 1.0, epsilon = 0.1)
clf = SVR(gamma = 'scale', C = 16.7, epsilon = 0.8)

clf.fit(X_train, y_train)

print(clf.predict(X_dev))

print(clf.score(X_dev, y_dev))



