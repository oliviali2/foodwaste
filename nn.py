import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

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



#mlp = MLPRegressor(hidden_layer_sizes = (20,), activation = 'relu', solver='adam',learning_rate='adaptive', max_iter=1000, learning_rate_init=0.05, alpha=0.01)
mlp = MLPRegressor(hidden_layer_sizes = (20,), activation = 'relu',
solver='adam',learning_rate='adaptive', max_iter=1000,
learning_rate_init=0.05, alpha=0.01, tol=1e-5, n_iter_no_change=15)

mlp.fit(X_train, y_train)
print "NN MSE: " + str(mean_squared_error(y_dev, mlp.predict(X_dev)))
print "NN R^2 score: " + str(mlp.score(X_dev, y_dev))



