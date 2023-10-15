from torch import nn
import torch


class RMSE(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		mse_loss_fn = nn.MSELoss()
		rmse = torch.sqrt(mse_loss_fn(x, y))
		return rmse
	
	@property
	def name(self):
		return 'Root Mean Square Error'
	

class MaxAbsoluteError(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		return torch.max(torch.abs(x - y))
	
	@property
	def name(self):
		return 'Max Absolute Error'