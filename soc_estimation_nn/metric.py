import torch
from abc import ABC, abstractmethod
from typing import Any
from torch.optim import Optimizer
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT


class Metric(ABC):
	
	@property
	@abstractmethod
	def name(self):
		raise NotImplementedError
	
	
class GradientNorms(Metric):


	@property
	def name(self):
		return 'GradientNorms'


	def compute_on_before_zero_grad(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
        optimizer: Optimizer
    ) -> None:

		return dict([
			( f'grad_norm({name})', float(torch.norm(parameter.grad)))  \
			for name, parameter in pl_module.model.named_parameters() if parameter.grad is not None
		])
	

class WeightsAndBiasesNorms(Metric):


	@property
	def name(self):
		return 'WeightsAndBiasesNorms'


	def compute_on_train_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:

		return dict([
			( f'norm({name})', float(torch.norm(parameter)))  \
			for name, parameter in pl_module.model.named_parameters() if parameter is not None
		])
    

class ValidationRMSE(Metric):


	@property
	def name(self):
		return 'ValidationRMSE'


	def compute_on_validation_batch_end(
		self, 
		trainer: pl.Trainer, 
		pl_module: pl.LightningModule, 
		outputs: STEP_OUTPUT | None, 
		batch: Any, 
		batch_idx: int, 
		dataloader_idx: int = 0
	) -> None:
		
		Y_pred = outputs.get('Y_pred')
		Y = outputs.get('Y')

		rmse = torch.sqrt(torch.mean((Y - Y_pred) ** 2))

		return {
			'rmse': float(rmse)
		}
	

class TrainingRMSE(Metric):


	@property
	def name(self):
		return 'TrainingRMSE'


	def compute_on_train_batch_end(
		self, 
		trainer: pl.Trainer, 
		pl_module: pl.LightningModule, 
		outputs: STEP_OUTPUT | None, 
		batch: Any, 
		batch_idx: int, 
		dataloader_idx: int = 0
	) -> None:
		
		Y_pred = outputs.get('Y_pred')
		Y = outputs.get('Y')

		rmse = torch.sqrt(torch.mean((Y - Y_pred) ** 2))

		return {
			'rmse': float(rmse)
		}
	


class ValidationMaxAbsoluteError(Metric):


	@property
	def name(self):
		return 'ValidationMaxAbsError'


	def compute_on_validation_batch_end(
		self, 
		trainer: pl.Trainer, 
		pl_module: pl.LightningModule, 
		outputs: STEP_OUTPUT | None, 
		batch: Any, 
		batch_idx: int, 
		dataloader_idx: int = 0
	) -> None:

		Y_pred = outputs.get('Y_pred')
		Y = outputs.get('Y')

		max_abs_error = torch.max(torch.abs(Y - Y_pred))

		return {
			'MaxAbsError': float(max_abs_error)
		}