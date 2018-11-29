from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

dataset = pd.read_csv('train_medium.csv')
enc = OneHotEncoder(handle_unknown = 'ignore')


X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

enc.fit(X)
print enc.categories_
onehotlabels = enc.transform(X).toarray()
onehotlabels.shape

X = onehotlabels


scaler = preprocessing.StandardScaler().fit(X) # Don't cheat - fit only on training data
X = scaler.transform(X) # apply same transformation to test data

X_train = X[:-50]
X_dev = X[-50:]

y_train = y[:-50]
y_dev = y[-50:]


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

prediction = clf.predict(X_dev)
print prediction

print y_dev
print mean_squared_error(y_dev, prediction)

