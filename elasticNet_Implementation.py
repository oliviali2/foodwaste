from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression

#X should be passed as fortran-contiguous numpy array

X, y = make_regression(n_features = 2, random_state = 0) #what is random state
regr = ElasticNet(random_state = 0)
regr.fit(X, y)
ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=0, selection='cyclic', tol=0.0001, warm_start=False)
print(regr.coef_)
print(regr.intercept_) 
print(regr.predict([[0, 0]])) 