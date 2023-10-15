import sys
sys.path.append('../')
from model_training.model import LSTM
from model_training.dataset import BatterySOCDataset
from soc_estimation_nn.loss import RMSE
from soc_estimation_nn.metric import GradientNorms, WeightsAndBiasesNorms
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
logger.log_info('Init training ...')


metrics = [
	GradientNorms(compute_on_before_zero_grad=True),
	WeightsAndBiasesNorms(compute_on_train_epoch_end=True)
]

for metric in metrics:
	logger.track(metric)
	logger.log_info(f'Tracking metric: {metric.name}')


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
	max_epochs=10, 
	logger=False,
	callbacks=[early_stop, logger],
)
logger.log_info('Starting training ...')

trainer.fit(
	training_module, 
	data_module, 
)	

logger.log_info('Training complete.')
