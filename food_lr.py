from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder 


import numpy as np
import pandas as pd

dataset = pd.read_csv('train_small.csv')
enc = OneHotEncoder(handle_unknown = 'ignore')


X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, 4].values
X_transform = enc.fit_transform(X)
#enc.fit(X)

X_train = X_transform[:-20]
X_test = X_transform[-20:]

Y_train = Y[:-20]
Y_test = Y[-20:]


# #logistic regression:
# clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, Y)
# clf.predict(X[:])
# clf.predict_proba(X[:])
# clf.score(X, Y)
# print clf.score(X,Y)



# #linear regression: 
# reg = linear_model.LinearRegression()
# reg.fit(X, Y)
# lr = linear_model.LinearRegression(copy_X = True, fit_intercept = True, n_jobs = None, normalize = False)
# print reg.coef_



regr = LinearRegression()

regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)

print Y_pred
print Y_test
# print 'Coefficient:  \n', regr.coef_
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_pred))

# print('Variance score: %.2f' % r2_score(Y_test, Y_pred))





#print("Mean squared error: %.2f" % np.mean((model.predict(X_transform) - Y) ** 2))



