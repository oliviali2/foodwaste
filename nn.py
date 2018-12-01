import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyplot 

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

numFolds = 2

#kfold splits your data into train and develop for you
kf = KFold(n_splits = numFolds)
kf.get_n_splits(X)
for train_index, dev_index in kf.split(X):
	X_train, X_dev = X[train_index], X[dev_index]
	y_train, y_dev = y[train_index], y[dev_index]


class Net(nn.Module):
	def__init__(self, name = 'model', input = 10):
		self.name = name
		self.net = torch.nn.Sequential(
			torch.nn.Lniear(input, 10)
			torch.nn.Relu()

			torch.nn.Linear(10,10)
			torch.nn.Relu()

			torch.nn.Linear(10, 1)
			torch.nn.Relu()
		)

	def forward(self, x):
		return self.net(x)

	def loss(self, output, y):
		return torch...MSE(output, y)


def train():
	model = NeuralNetwork('our_model')
	optimizer = torch.optim.Adam(...)

	for e in range(epochs):
		for x, y in data:
			optimizer.zero_grad()

			output = model(x)
			loss = model.loss(output, y)

			loss.backward()
			optimizer.step()

