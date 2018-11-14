import numpy as py
#import sys
import numpy as np
import pandas as pd
#import itertools
import torch
import torch.nn as nn
import math, random

n_in, n_h, n_out, batch_size = 10, 5, 1, 10 #define sizes of all the layers and batch size


# import matplotlib
# import collections
# import random
# import operator
# import timei

def main(): 
	print "start"
	inputfilename = "train_small.csv"

	data = pd.read_csv(inputfilename)	

	print "finished"

	x = torch.randn(batch_size, n_in)
	y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
	model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	for epoch in range(50):
		#Forward Propagation
		y_pred = model(x)

    	# print 'epoch: '
    	print "hi"

    	loss = criterion(y_pred, y)

    	print "hi2"
    	print('epoch: ', epoch,' loss: ', loss.item())
   		# Zero the gradients
    	optimizer.zero_grad()
    
    	# perform a backward pass (backpropagation)
    	loss.backward()
    
    	# Update the parameters
    	optimizer.step()




if __name__ == '__main__':
    main()
