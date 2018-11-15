from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#ID, date, store number, item number, unit sales, on promotion

plt.ion()   # interactive mode

data = pd.read_csv('train_small.csv')

class dataSet(dataset):
	def __init__(self, csv_file):
		self.csv = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.csv)

    


