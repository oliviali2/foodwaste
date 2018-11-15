from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


#fix random seed for reproducibility
seed = 7

#dataset = np.loadtxt("train_small_no_label.csv", delimiter=",")
dataset = pd.read_csv('train_small.csv')

#split into input (X) and output (Y) variables

X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, 4].values

Y = np.reshape(Y, (-1, 1))
scaler = MinMaxScaler()
print(scaler.fit(X))
print(scaler.fit(Y))

xscale = scaler.transform(X)
yscale = scaler.transform(Y)

#define model
def baseline_model():
	model = Sequential()

	model.add(Dense(3, input_dim = 3, kernel_initializer = 'normal', activation = 'relu')) #number of neurons in input layer?
	#model.add(Dense(3, kernel_initializer = 'normal', activation = 'relu')) #number of neurons in hidden layer????
	#model.add(Dense(1, activation = 'sigmoid'))

	model.add(Dense(1, kernel_initializer = 'normal'))

	#compile model
	#[TODO] think about which loss function to use --- mse? mae? etc.
	model.compile(loss='mean_squared_error', optimizer='adam') 
	return model

def larger_model():
	model = Sequential()

	model.add(Dense(6, input_dim = 3, kernel_initializer = 'normal', activation = 'relu')) #number of neurons in input layer?
	model.add(Dense(3, kernel_initializer = 'normal', activation = 'relu')) #number of neurons in hidden layer????

	model.add(Dense(1, kernel_initializer = 'normal'))

	#compile model
	#[TODO] think about which loss function to use --- mse? mae? etc.
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse', 'mae']) 
	return model


def wider_model(): #1.06
	model = Sequential()

	model.add(Dense(8, input_dim = 3, kernel_initializer = 'normal', activation = 'relu')) #number of neurons in input layer?
	
	#model.add(Dense(1, activation = 'sigmoid'))

	model.add(Dense(1, kernel_initializer = 'normal'))

	#compile model
	#[TODO] think about which loss function to use --- mse? mae? etc.
	model.compile(loss='mean_squared_error', optimizer='adam') 
	return model

#fit the model
#[TODO] play with epoch and batch size
#model.fit(X, Y, epochs = 150, batch_size = 5, verbose = 0)


#BASELINE
# estimator = KerasRegressor(build_fn = baseline_model, epochs = 100, batch_size = 5, verbose = 0)

# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# #evaluate model
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# predictions = model.predict(X)
# rounded = [round(x[0]) for x in predictions]
# print(rounded)

#history = larger_model().fit(xscale, yscale, epochs=150, batch_size=10,  verbose=0)

# evaluate model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MAE" % (results.mean(), results.std()))
print results


