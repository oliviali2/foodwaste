import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyplot as plt

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

class LinearRegressionModel(nn.Module):
	def__init__(self, input_dim, output_dim):
		super(LinearRegressionModel, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		out = self.linear(x)
		return out

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
l_rate = 0.01
optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) #stochastic Gradient Descent
epochs = 2000

for epoch in range(epochs):
	inputs = Variable(torch.from_numpy(x_train))
	labels = Variable(torch.from_numpy(y_correct))
	optimiser.zero_grad()
	outputs = model.forward(inputs)
	loss = criterion(outputs, labels)
	loss.backward()
	optimiser.step()
	print('epoch {}, loss {}'.format(epoch, loss.data[0]))

predicted = model.forward(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_correct, 'go', label = 'from data', alpha = 0.5)
plt.plot(x_train, predicted, labl = 'prediction', alpha = 0.5)
plt.legend()
plt.show()
print(model.state_dict())
