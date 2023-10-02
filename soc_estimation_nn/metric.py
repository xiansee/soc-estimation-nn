from torchmetrics import Metric
from torch import nn, Tensor
import torch


class RMSELoss(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		mse_loss_fn = nn.MSELoss()
		rmse_loss = torch.sqrt(mse_loss_fn(x, y))
		return rmse_loss
	
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
	

class TensorNorm(Metric):

	full_state_update: bool = False
    
	def __init__(self):
		super().__init__()
		self.add_state('value', default=torch.tensor(0), dist_reduce_fx='mean')

	def update(self, gradient_tensor: Tensor):
		self.value = torch.norm(gradient_tensor)

	def compute(self):
		return self.value.float()
    