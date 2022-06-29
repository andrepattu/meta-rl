import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		# THESE HYPERPARAMETERS ARE THE ONES THAT PRODUCED THE CURRENT BASELINE
		# self.layer1 = nn.Linear(in_dim, 64)
		# self.layer2 = nn.Linear(64, 64)
		# self.layer3 = nn.Linear(64, out_dim)
		
		# THESE HYPERPARAMETERS ARE SUGGESTED AFTER REFACTORING TO IMPROVE THE BASELINE PERFORMANCE
		self.layer1 = nn.Linear(in_dim, 128)
		self.layer2 = nn.Linear(128, 128)
		self.layer3 = nn.Linear(128, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				obs - observation to pass as input
			Return:
				output - the output of the forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float).to(self.device)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output