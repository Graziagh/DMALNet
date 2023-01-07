import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pdb

class COVID19(Dataset):
	def __init__(self, data_dir = './', filename = 'train',  transform=None):
		super().__init__()

		dataset = np.load(data_dir+filename+'.npz')
		self.data = dataset['train']
		self.targets = dataset['label']
		self.transform = transform
		self.targets = torch.LongTensor(self.targets)

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, index):

		image, label = self.data[index], self.targets[index]
		if self.transform is not None:
			image = self.transform(image)
		return image, label
