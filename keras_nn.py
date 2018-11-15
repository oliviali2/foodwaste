import numpy as py
#import sys
import numpy as np
import pandas as pd
#import itertools
import torch
import torch.nn as nn
import math, random




# import matplotlib
# import collections
# import random
# import operator
# import timei

def main(): 

	inputfilename = "train_small.csv"

	data = pd.read_csv(inputfilename)	


	n_in, n_h, n_out, batch_size = 1, 1, 1, 10 #define sizes of all the layers and batch size
	x = torch.randn(batch_size, n_in)
	y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

	model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	for epoch in range(100):
		y_pred = model(x)
		loss = criterion(y_pred, y)
		print('epoch: ', epoch,' loss: ', loss.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()



if __name__ == '__main__':
    main()
