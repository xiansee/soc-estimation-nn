import sys
sys.path.append('../')
from model_training.model import LSTM
from model_training.dataset import BatterySOCDataset
from soc_estimation_nn.loss import RMSE
from soc_estimation_nn.metric import GradientNorms, WeightsAndBiasesNorms, ValidationRMSE, TrainingRMSE, ValidationMaxAbsoluteError
from soc_estimation_nn.training import TrainingModule
from soc_estimation_nn.logger import TrainingLogger
from soc_estimation_nn.data_module import DataModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl
from datetime import datetime
 

dataset = BatterySOCDataset(data_directory='../data/processed')
model = LSTM(
	input_size=3, 
	hidden_size=50,
	output_size=1, 
	num_lstm_layers=1
)

root_directory = f'../model_training/{datetime.utcnow().strftime("%Y_%m_%d_T%H_%M_%SZ")}'
data_module =  DataModule(dataset=dataset)
logger = TrainingLogger(
	training_name='LSTM',
	log_directory=root_directory,
	stream_log=True
)

for metric in [
	GradientNorms(),
	WeightsAndBiasesNorms(),
	ValidationRMSE(),
	TrainingRMSE(),
	ValidationMaxAbsoluteError(),
]:
	logger.track_metric(metric)
	
early_stop = EarlyStopping(
	monitor='validation_accuracy', 
	min_delta=0.001, 
	patience=5, 
	mode='min', 
	stopping_threshold=0.01, 
	check_on_train_epoch_end=True
)

training_module = TrainingModule(
	model=model, 
	training_logger=logger,
	loss_function=RMSE(),
	initial_lr=0.01,
	weight_decay=0.002
)

trainer = pl.Trainer(
	default_root_dir=root_directory,
	max_epochs=5, 
	logger=False,
	callbacks=[early_stop, logger],
	enable_progress_bar=False,
	enable_model_summary=False,
	enable_checkpointing=True,
)

import time

start_time = time.time()

trainer.fit(
	training_module, 
	data_module, 
)	

print(f'Total time: {time.time() - start_time}')

