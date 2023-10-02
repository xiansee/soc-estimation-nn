from torch import nn, optim
import lightning.pytorch as pl
from soc_estimation_nn.metric import TensorNorm


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


	def log_gradient_norms(self):
		if not hasattr(self, 'gradient_norm_metric'):
			self.gradient_norm_metric = dict([
				(name, TensorNorm())  \
				for name, _ in self.model.named_parameters()
			])
			
		gradient_norms = dict([
			(f'grad_norm({name})', self.gradient_norm_metric.get(name)(parameter.grad))  \
			for name, parameter in self.model.named_parameters() if parameter.grad is not None
		])
		self.log_dict(gradient_norms, on_epoch=True, on_step=False)


	def log_weights_and_biases_norms(self):
		if not hasattr(self, 'weights_and_biases_norm_metric'):
			self.weights_and_biases_norm_metric = dict([
				(name, TensorNorm())  \
				for name, _ in self.model.named_parameters()
			])

		weight_and_biases_norm = dict([
			(f'norm({name})', self.weights_and_biases_norm_metric.get(name)(parameter))  \
			for name, parameter in self.model.named_parameters() if parameter is not None
		])
		self.log_dict(weight_and_biases_norm, on_epoch=True, on_step=False)


	def training_step(
		self, 
		batch, 
		batch_idx
	):
		X, SOC = batch
		SOC_pred = self.model(X)
		loss = self.loss_fn(SOC, SOC_pred)
		
		self.log_gradient_norms()
		self.log_weights_and_biases_norms()

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
