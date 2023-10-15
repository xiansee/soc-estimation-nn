from torch import nn, optim
import lightning.pytorch as pl
from soc_estimation_nn.logger import TrainingLogger


class TrainingModule(pl.LightningModule):


	def __init__(
		self, 
		model: nn.Module, 
		training_logger: TrainingLogger,
		loss_function: nn.Module,
		initial_lr: float = 0.01,
		weight_decay: float = 0.001
	):
		super().__init__()
		self.model = model
		self.training_logger = training_logger
		self.loss_fn = loss_function
		self.initial_lr = initial_lr
		self.weight_decay = weight_decay


	def training_step(
		self, 
		batch, 
		batch_idx
	):
		X, SOC = batch
		SOC_pred = self.model(X)
		loss = self.loss_fn(SOC, SOC_pred)

		return loss
	

	def validation_step(
		self, 
		batch, 
		batch_idx
	):
		X, SOC = batch
		SOC_pred = self.model(X)
		validation_accuracy = self.loss_fn(SOC, SOC_pred)
		self.log('validation_accuracy', validation_accuracy, on_epoch=True, on_step=False, prog_bar=True)
	

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
		return optimizer
