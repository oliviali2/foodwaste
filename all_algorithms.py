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
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

# dataset = pd.read_csv('data/train_medium.csv')
#dataset = pd.read_csv('data/cleaned_train_medium.csv')
dataset = pd.read_csv('data/cleaned_train_v5.csv')
has_n = dataset.columns[dataset.isna().any()].tolist()
print('these cols have nan:', has_n)

enc = OneHotEncoder(handle_unknown = 'ignore')

X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values

ID = dataset.iloc[:, 3].values


enc.fit(X)
#print enc.categories_
onehotlabels = enc.transform(X).toarray()
onehotlabels.shape

X = onehotlabels

# scaler = preprocessing.StandardScaler().fit(X)
# X = scaler.transform(X) 

numFolds = 2

#kfold splits your data into train and develop for you
kf = KFold(n_splits = numFolds)
# kf = RepeatedKFold(n_splits = numFolds, n_repeats = 10, random_state = None)

for train_index, dev_index, in kf.split(X):
	X_train, X_dev = X[train_index], X[dev_index]
	y_train, y_dev, y_ID = y[train_index], y[dev_index], ID[dev_index]

print y_ID

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) 
X_dev = scaler.transform(X_dev)

# print len(X_train)
# print len(X_dev)


difs = []
labels = []



def oracle_mean():
	y_pred = []
	for i in y_ID:
		if i == 265559: y_pred.append(49)
		elif i == 314384: y_pred.append(41)
		elif i == 364606: y_pred.append(52)
		elif i == 502331: y_pred.append(44)
		elif i == 819932: y_pred.append(59)
		elif i == 1047679: y_pred.append(65)
		elif i == 1463992: y_pred.append(91)
		elif i == 1473474: y_pred.append(106)
		elif i == 1503844: y_pred.append(220)
		elif i == 807493: y_pred.append(55)

	print ("mean oracle score: " + str(r2_score(y_dev, y_pred)))

def oracle_med():
	y_pred = []
	for i in y_ID:
		if i == 265559: y_pred.append(37)
		elif i == 314384: y_pred.append(32)
		elif i == 364606: y_pred.append(42)
		elif i == 502331: y_pred.append(38)
		elif i == 819932: y_pred.append(38)
		elif i == 1047679: y_pred.append(24)
		elif i == 1463992: y_pred.append(50)
		elif i == 1473474: y_pred.append(63)
		elif i == 1503844: y_pred.append(154)
		elif i == 807493: y_pred.append(33)

	print ("med oracle score: " + str(r2_score(y_dev, y_pred)))

def run_LR():
	regr = LinearRegression(normalize = True)
	regr.fit(X_train, y_train)

	y_predict = regr.predict(X_dev)
	set_lower_bound(y_predict, max(y_train), min(y_train))

	print ("LR MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print("LR R^2 score: " + str(regr.score(X_dev, y_dev)))

	test_on_min_max(y_predict, y_dev)

	#plot("LR", X_dev, y_dev, y_predict)

def run_SGDR():
	clf = linear_model.SGDRegressor(max_iter=100, tol=1e-3, learning_rate = 'adaptive')
	clf.fit(X_train, y_train)

	y_predict = clf.predict(X_dev)
	set_lower_bound(y_predict, max(y_train), min(y_train))

	print("SGDR MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print("SGDR R^2 score: " + str(clf.score(X_dev, y_dev)))

	test_on_min_max(y_predict, y_dev)
	#plot("SGDR", X_dev, y_dev, y_predict)

def run_Lasso():
	a = 1e-10
	lassoreg = Lasso(alpha=1.75, random_state=1, max_iter=100, copy_X = True)
	lassoreg.fit(X_train,y_train)
	y_predict = lassoreg.predict(X_dev)

	print("Lasso MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print("Lasso R^2 score: " + str(lassoreg.score(X_dev, y_dev)))

	test_on_min_max(y_predict, y_dev)
	#plot("Lasso", X_dev, y_dev, y_predict)

def run_SVR():
	clf = SVR(gamma = 'scale', C = 50, epsilon = 0.8)
	#clf = SVR(gamma = 'scale', C = 16.7, epsilon = e)
	clf.fit(X_train, y_train)

	y_predict = clf.predict(X_dev)

	set_lower_bound(y_predict, max(y_train), min(y_train))

	print ("SVR MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print("SVR R^2 score: " + str(clf.score(X_dev, y_dev)))

	test_on_min_max(y_predict, y_dev)
	#plot("SVR", X_dev, y_dev, y_predict)


def run_NN():
	# global X_train, X_dev
	#mlp = MLPRegressor(hidden_layer_sizes = (3,), activation = 'relu', solver='adam',learning_rate='adaptive', max_iter=1000, learning_rate_init=0.01, alpha=0.01)
	# mlp = MLPRegressor(hidden_layer_sizes = (20,), activation = 'relu',
	# solver='adam',learning_rate='adaptive', max_iter=1000,
	# learning_rate_init=0.05, alpha=0.01, tol=1e-5, n_iter_no_change=15, random_state = 1)
	
	# scaler = preprocessing.StandardScaler().fit(X_train)
	# X_train = scaler.transform(X_train)
	# X_dev = scaler.transform(X_dev) 
	mlp = MLPRegressor(hidden_layer_sizes = (9), activation = 'relu', 
		solver='adam', max_iter=1500, learning_rate_init=0.05, alpha=0.04, 
		tol=1e-4, n_iter_no_change=1000, random_state=1, early_stopping=True, 
		beta_1 = 0.31, beta_2 = 0.98)
	mlp.fit(X_train, y_train)

	y_predict = mlp.predict(X_dev)
	set_lower_bound(y_predict, max(y_train), min(y_train))

	print( "NN MSE: " + str(mean_squared_error(y_dev, y_predict)))
	print( "NN R^2 score: " + str(mlp.score(X_dev, y_dev)))

	# y_train_predict = mlp.predict(X_train)
	# set_lower_bound(y_train_predict)

	# print( "NN train R^2 score: " + str(mlp.score(X_train, y_train_predict)))


	test_on_min_max(y_predict, y_dev)
	#plot("NN", X_dev, y_dev, y_predict)

def set_lower_bound(y_predict, ma, mi):
	for idx, val in enumerate(y_predict):
		break
		#y_predict[idx] = max(mi, val)
		#y_predict[idx] = min(ma, val)


def plot(title, X_train, y_train, y_predict):
	X_revert = enc.inverse_transform(X_train)
	# print X_revert
	# print X_revert[:,2]

	X_plot = np.array(X_revert[:,2])
	y_plot = np.array(y_train)
	predict_plot = np.array(y_predict)

	plt.title(title)
	dif = (predict_plot - y_plot)

	difs.append(dif)
	labels.append(title)

	print float(sum(dif)) / float(len(y_plot))
	zeros = listofzeros = [0] * len(dif)
	plt.plot(dif, color = 'g')
	plt.plot(zeros, color = 'b')

	# plt.plot(y_plot, color = 'g')
	# plt.plot(predict_plot, color = 'b')

	# plt.xlim(-10, len(y_plot) + 10)
	#plt.ylim(min(dif) - 100, max(dif) + 100)

	# plt.ylim(-500, 500)
	# plt.xlim(0, len(dif))

	plt.autoscale(axis = 'y')

	axes = plt.gca()
	#axes.set_ylim([min(dif) - 5, max(dif) + 5])
	#axes.set_xlim([-10, len(dif) + 10])

	plt.xlabel("Index")
	plt.ylabel("Unit Sales")

	# plt.scatter(X_plot, y_plot, color='g')
	# plt.plot(X_plot, predict_plot,color='k')

	plt.show()

def plot_all(title, X_train, y_train):
	global difs, labels
	X_revert = enc.inverse_transform(X_train)
	# print X_revert
	# print X_revert[:,2]

	X_plot = np.array(X_revert[:,2])
	y_plot = np.array(y_train)
	#predict_plot = np.array(y_predict)
	plt.ylim(-500, 500)
	plt.xlim(0, len(difs[0]))
	plt.title(title)

	for i in range(len(difs)): 
		dif = difs[i]
		dif.sort()
		plt.plot(dif, label = labels[i])
	zeros = listofzeros = [0] * len(y_plot)

	plt.plot(zeros)
	plt.xlabel("Index")
	plt.ylabel("Unit Sales")

	plt.legend()
	plt.show()

def test_on_min_max(y_pred, y_dev):
	max_pred = []
	max_dev = []
	med_pred = []
	med_dev = []
	min_pred = []
	min_dev = []


	for idx, i in enumerate(y_ID):
		if i == 265559:
			max_pred.append(y_pred[idx])
			max_dev.append(y_dev[idx])
		elif i == 1503844: 
			min_pred.append(y_pred[idx])
			min_dev.append(y_dev[idx])
		elif i == 364606:
			med_pred.append(y_pred[idx])
			med_dev.append(y_dev[idx])

	print len(max_pred)
	print len(med_pred)
	print len(min_pred)
	print "max EVS: " + str( explained_variance_score(max_dev, max_pred) )
	print "med EVS: " + str( explained_variance_score(med_dev, med_pred) )
	print "min EVS: " + str(explained_variance_score(min_dev, min_pred))
	print "\n"
	#print ("max_val score: " + str(r2_score(max_dev, max_pred)))
	#print ("min_val score: " + str(r2_score(min_dev, min_pred)))

def main():
	#oracle_mean()
	#oracle_med()
	run_LR()
	run_SGDR()
	run_Lasso()
	#run_SVR()
	#run_NN()

	#plot_all("All Algorithms", X_train, y_train)




main()

