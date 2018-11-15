from keras.models import Sequential
from keras.layers import Dense
import numpy

#fix random seed for reproducibility
numpy.random.seed(7)

dataset = numpy.loadtxt("train_small_no_label.csv", delimiter=",")
#split into input (X) and output (Y) variables

X = dataset[:, 1:4]
Y = dataset[:, 4]

#define model
model = Sequential()
model.add(Dense(3, input_dim = 3, activation = 'relu')) #number of neurons in input layer?
model.add(Dense(3, activation = 'relu')) #number of neurons in hidden layer????
model.add(Dense(1, activation = 'sigmoid'))

#compile model
#[TODO] think about which loss function to use --- mse? mae? etc.
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error']) 

#fit the model
#[TODO] play with epoch and batch size
model.fit(X, Y, epochs = 150, batch_size = 5)

#evaluate model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
rounded = [(roundedx[0]) for x in predictions]
print(rounded)

