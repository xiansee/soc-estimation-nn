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
import optuna
 

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

logger.track_metrics([
	GradientNorms(),
	WeightsAndBiasesNorms(),
	ValidationRMSE(),
	TrainingRMSE(),
	ValidationMaxAbsoluteError(),
])
	
early_stop = EarlyStopping(
	monitor='validation_accuracy', 
	min_delta=0.001, 
	patience=5, 
	mode='min', 
	stopping_threshold=0.01, 
	check_on_train_epoch_end=True
)

def define_trainer(trial):
	lr = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
	wd = trial.suggest_float('weight_decay', 1e-3, 1e-2)

	training_module = TrainingModule(
		model=model, 
		loss_function=RMSE(),
		initial_lr=lr,
		weight_decay=wd
	)

	return training_module

def objective(trial):
	logger.trial_number = trial.number
	
	training_module = define_trainer(trial)	
	trainer = pl.Trainer(
		default_root_dir=root_directory,
		max_epochs=50, 
		logger=False,
		callbacks=[early_stop, logger],
		enable_progress_bar=False,
		enable_model_summary=False,
		enable_checkpointing=True,
	)

	logger.log_hyperparameters(
		trial_params=trial.params
	)

	trainer.fit(
		training_module, 
		data_module, 
	)	
	
	logger.log_trial(
		trial=trial, 
		trainer_metrics=trainer.logged_metrics
	)

	validation_accuracy = trainer.logged_metrics.get('validation_accuracy')
	return validation_accuracy

study = optuna.create_study()
study.optimize(objective, n_trials=10)

logger.log_best_hyperparams(
	study=study
)
