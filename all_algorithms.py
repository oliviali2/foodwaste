'''
This code contains all the models and algorithms used for this project:
1. SVR
2. SGD
3. Linear regression
4. Neural network
'''

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
from sklearn.linear_model import Lasso

# dataset = pd.read_csv('data/train_medium.csv')
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

def run_Lasso():
	a = 1e-10
	lassoreg = Lasso(alpha=a,normalize=True, max_iter=1e5)
	lassoreg.fit(X_train,y_train)
	y_predict = lassoreg.predict(X_train)

	print("Lasso MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print("Lasso R^2 score: " + str(lassoreg.score(X_dev, y_dev)))

	plot("Lasso", X_dev, y_dev, y_predict)

def run_SVR():
	clf = SVR(gamma = 'scale', C = 16.7, epsilon = 0.8)
	clf.fit(X_train, y_train)

	y_predict = clf.predict(X_dev)
	set_lower_bound(y_predict)

	print ("SVR MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print("SVR R^2 score: " + str(clf.score(X_dev, y_dev)))

	plot("SVR", X_dev, y_dev, y_predict)


def run_SGDR():
	clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
	clf.fit(X_train, y_train)

	y_predict = clf.predict(X_dev)
	set_lower_bound(y_predict)

	print("SGDR MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print("SGDR R^2 score: " + str(clf.score(X_dev, y_dev)))


	plot("SGDR", X_dev, y_dev, y_predict)

def run_LR():
	regr = LinearRegression(normalize = True)
	regr.fit(X_train, y_train)

	y_predict = regr.predict(X_dev)
	set_lower_bound(y_predict)

	print ("LR MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print("LR R^2 score: " + str(regr.score(X_dev, y_dev)))

	plot("LR", X_dev, y_dev, y_predict)

def run_NN():
	#mlp = MLPRegressor(hidden_layer_sizes = (3,), activation = 'relu', solver='adam',learning_rate='adaptive', max_iter=1000, learning_rate_init=0.01, alpha=0.01)
	mlp = MLPRegressor(hidden_layer_sizes = (20,), activation = 'relu',
	solver='adam',learning_rate='adaptive', max_iter=1000,
	learning_rate_init=0.05, alpha=0.01, tol=1e-5, n_iter_no_change=15)
	mlp.fit(X_train, y_train)

	y_predict = mlp.predict(X_dev)
	set_lower_bound(y_predict)

	print( "NN MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print( "NN R^2 score: " + str(mlp.score(X_dev, y_dev)))

	plot("NN", X_dev, y_dev, y_predict)

def set_lower_bound(y_predict):
	for idx, val in enumerate(y_predict):
		y_predict[idx] = max(0, val)


def plot(title, X_train, y_train, y_predict):
	X_revert = enc.inverse_transform(X_train)
	# print X_revert
	# print X_revert[:,2]
	X_plot = np.array(X_revert[:,2])
	y_plot = np.array(y_train)
	predict_plot = np.array(y_predict)

	plt.title(title)
	plt.plot(y_plot, color = 'g')
	plt.plot(predict_plot, color = 'b')
	axes = plt.gca()
	axes.set_ylim([-5,max(max(y_plot), max(predict_plot)) + 5])
	axes.set_xlim([-100,len(y_plot) + 100])
	plt.xlabel("Index")
	plt.ylabel("Unit Sales")

	# plt.scatter(X_plot, y_plot, color='g')
	# plt.plot(X_plot, predict_plot,color='k')

	plt.show()


def main():
	run_LR()
	run_SGDR()
	run_SVR()
	run_NN()
	run_Lasso()

main()
