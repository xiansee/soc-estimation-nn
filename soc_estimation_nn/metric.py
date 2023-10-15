import torch
from abc import ABC, abstractmethod


class Metric(ABC):


	def __init__(
		self,
		compute_on_before_zero_grad: bool = False,
		compute_on_train_epoch_end: bool = False,
		compute_on_validation_epoch_end: bool = False,
	):
		self._compute_on_before_zero_grad = compute_on_before_zero_grad
		self._compute_on_train_epoch_end = compute_on_train_epoch_end
		self._compute_on_validation_epoch_end = compute_on_validation_epoch_end

	
	@property
	@abstractmethod
	def name(self):
		raise NotImplementedError


	@abstractmethod
	def compute(self):
		raise NotImplementedError


	@property
	def compute_on_before_zero_grad(self):
		return self._compute_on_before_zero_grad
	
	
	@property
	def compute_on_train_epoch_end(self):
		return self._compute_on_train_epoch_end
    

	@property
	def compute_on_validation_epoch_end(self):
		return self._compute_on_validation_epoch_end
	
	
class GradientNorms(Metric):


	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	@property
	def name(self):
		return 'GradientNorms'


	def compute(
		self, 
		trainer, 
		training_module
	) -> dict[str, float]:

		return dict([
			( f'grad_norm({name})', float(torch.norm(parameter.grad)))  \
			for name, parameter in training_module.model.named_parameters() if parameter.grad is not None
		])
	

class WeightsAndBiasesNorms(Metric):


	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	@property
	def name(self):
		return 'WeightsAndBiasesNorms'


	def compute(
		self, 
		trainer, 
		training_module
	) -> dict[str, float]:

		return dict([
			( f'norm({name})', float(torch.norm(parameter)))  \
			for name, parameter in training_module.model.named_parameters() if parameter is not None
		])
    