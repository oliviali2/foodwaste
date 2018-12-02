#all algorithms
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# dataset = pd.read_csv('data/train_medium.csv')
dataset = pd.read_csv('data/cleaned_train_medium.csv')
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

numFolds = 2

#kfold splits your data into train and develop for you
kf = KFold(n_splits = numFolds)
kf.get_n_splits(X)
for train_index, dev_index in kf.split(X):
	X_train, X_dev = X[train_index], X[dev_index]
	y_train, y_dev = y[train_index], y[dev_index]


def run_SVR():
	clf = SVR(gamma = 'scale', C = 1.0, epsilon = 0.2)
	clf.fit(X_train, y_train)
	print "SVR MSE: " + str(mean_squared_error(y_dev, clf.predict(X_dev)))
	print("SVR R^2 score: " + str(clf.score(X_dev, y_dev)))

def run_SGD():
	clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
	clf.fit(X_train, y_train)

	print "SGD MSE: " + str(mean_squared_error(y_dev, clf.predict(X_dev)))
	print("SGD R^2 score: " + str(clf.score(X_dev, y_dev)))

def run_LR():
	regr = LinearRegression()
	regr.fit(X_train, y_train)
	print "LR MSE: " + str(mean_squared_error(y_dev, regr.predict(X_dev)))
	print("LR R^2 score: " + str(regr.score(X_dev, y_dev)))

def run_NN():
	mlp = MLPRegressor(hidden_layer_sizes = (3,), activation = 'relu', solver='adam',learning_rate='adaptive', max_iter=1000, learning_rate_init=0.01, alpha=0.01)

	mlp.fit(X_train, y_train)
	print "NN MSE: " + str(mean_squared_error(y_dev, mlp.predict(X_dev)))
	print "NN R^2 score: " + str(mlp.score(X_dev, y_dev))

def main():
	run_SVR()
	run_SGD()
	run_LR()
	run_NN()

main()
