import sys
sys.path.append('../')
from soc_estimation_nn.model import SOCEstNN
from soc_estimation_nn.metric import RMSELoss
from soc_estimation_nn.training import TrainingModule
from soc_estimation_nn.dataset import BatterySOCDataset
from soc_estimation_nn.helper import plot
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


if __name__ == '__main__':

	tsd = BatterySOCDataset('../data/processed')
	model = SOCEstNN(
		input_size=3, 
		hidden_size=50,
		output_size=1, 
		num_lstm_layers=1
	)
	
	training_set, validation_set = random_split(tsd, [10, 3])
	training_dataloader = DataLoader(training_set, batch_size=1, shuffle=True)
	validation_dataloader = DataLoader(validation_set, batch_size=1)

	early_stop_cellback = EarlyStopping(
		monitor='validation_accuracy', 
		min_delta=0.001, 
		patience=5, 
		mode='min', 
		stopping_threshold=0.01, 
		check_on_train_epoch_end=True
	)
	model_training = TrainingModule(
		model=model, 
		loss_function=RMSELoss(),
		initial_lr=0.01,
		weight_decay=0.002
	)
	
	trainer = pl.Trainer(
		max_epochs=500, 
		log_every_n_steps=len(tsd), 
		# callbacks=[early_stop_cellback]
	)
	trainer.fit(
		model=model_training, 
		train_dataloaders=training_dataloader, 
		val_dataloaders=validation_dataloader
	)	
