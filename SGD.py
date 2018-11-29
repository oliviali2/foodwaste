from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

dataset = pd.read_csv('train_medium.csv')
enc = OneHotEncoder(handle_unknown = 'ignore')


X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

enc.fit(X)
#print enc.categories_
onehotlabels = enc.transform(X).toarray()
onehotlabels.shape

X = onehotlabels


scaler = preprocessing.StandardScaler().fit(X) # Don't cheat - fit only on training data
X = scaler.transform(X) # apply same transformation to test data

# X_train = X[:-50]
# X_dev = X[-50:]
# y_train = y[:-50]
# y_dev = y[-50:]

numFolds = 2

#kfold splits your data into train and develop for you
kf = KFold(n_splits = numFolds)
kf.get_n_splits(X)
for train_index, dev_index in kf.split(X):
	X_train, X_dev = X[train_index], X[dev_index]
	y_train, y_dev = y[train_index], y[dev_index]


clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)

# linear_model.SGDRegressor(alpha=0.0001, average=False, early_stopping=False,
#        epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
#        learning_rate='invscaling', loss='squared_loss', max_iter=1000,
#        n_iter=None, n_iter_no_change=5, penalty='l2', power_t=0.25,
#        random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
#        verbose=0, warm_start=False)

trainPrediction = clf.predict(X_train)
print mean_squared_error(y_train, trainPrediction)
# print trainPrediction
# print y_train


prediction = clf.predict(X_dev)
# print prediction
# print y_dev
print mean_squared_error(y_dev, prediction)


# seed = 7
# kfold = model_selection.KFold(n_splits=2, random_state=seed)
# model = linear_model.SGDRegressor()
# scoring = 'neg_mean_absolute_error'
# results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
# print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
