import lightning.pytorch as pl
from torch import nn, optim

from soc_estimation_nn.logger import TrainingLogger


class TrainingModule(pl.LightningModule):


	def __init__(
		self, 
		model: nn.Module, 
		loss_function: nn.Module,
		initial_lr: float = 0.01,
		weight_decay: float = 0.001
	):
		super().__init__()
		self.model = model
		self.loss_fn = loss_function
		self.initial_lr = initial_lr
		self.weight_decay = weight_decay


	def training_step(
		self, 
		batch, 
		batch_idx
	):
		X, Y = batch
		Y_pred = self.model(X)
		training_loss = self.loss_fn(Y, Y_pred)

		self.log(
			'training_loss', 
			training_loss, 
			on_epoch=True, 
			on_step=False, 
			prog_bar=True
		)

		return {
			'loss': training_loss,
			'Y': Y,
			'Y_pred': Y_pred
		}
	

	def validation_step(
		self, 
		batch, 
		batch_idx
	):
		X, Y = batch
		Y_pred = self.model(X)
		validation_accuracy = self.loss_fn(Y, Y_pred)

		self.log(
			'validation_accuracy', 
			validation_accuracy, 
			on_epoch=True, 
			on_step=False, 
			prog_bar=True
		)

		return {
			'loss': validation_accuracy,
			'Y': Y,
			'Y_pred': Y_pred
		}
	
		

	def configure_optimizers(self):
		optimizer = optim.Adam(
			self.parameters(), 
			lr=self.initial_lr, 
			weight_decay=self.weight_decay
		)

		return optimizer
