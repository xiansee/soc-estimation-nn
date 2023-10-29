import sys
sys.path.append('../')
from model_training.model.model_v1 import ModelV1
from model_training.model.model_v2 import ModelV2
from model_training.dataset.dataset_v1 import DatasetV1
from model_training.dataset.dataset_v2 import DatasetV2
from model_training.dataset.dataset_v3 import DatasetV3
from soc_estimation_nn.loss import RMSE
from soc_estimation_nn.metric import GradientNorms, WeightsAndBiasesNorms, ValidationRMSE, TrainingRMSE, ValidationMaxAbsoluteError
from soc_estimation_nn.training import TrainingModule
from soc_estimation_nn.logger import TrainingLogger
from soc_estimation_nn.data_module import DataModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl
from datetime import datetime
import optuna
 

dataset = DatasetV2(data_directory='../data/processed')

root_directory = f'../model_training/training_logs/{datetime.utcnow().strftime("%Y_%m_%d_T%H_%M_%SZ")}'
data_module =  DataModule(dataset=dataset)
logger = TrainingLogger(
	training_name='LSTM',
	log_directory=root_directory,
	stream_log=True
)

logger.log_info('ModelV1')
logger.log_info('DatasetV3')

logger.track_metrics([
	GradientNorms(),
	WeightsAndBiasesNorms(),
	ValidationRMSE(),
	TrainingRMSE(),
	ValidationMaxAbsoluteError(),
])
	
def define_model(trial):
	hidden_size = trial.suggest_int('lstm_hidden_size', 10, 100)
	num_of_layers = trial.suggest_int('lstm_layers', 1, 3)

	model = ModelV1(
		input_size=3, 
		hidden_size=hidden_size,
		output_size=1, 
		num_lstm_layers=num_of_layers,
	)
	return model

def define_trainer(trial):
	learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
	weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-2)

	model = define_model(trial)
	training_module = TrainingModule(
		model=model, 
		loss_function=RMSE(),
		initial_lr=learning_rate,
		weight_decay=weight_decay
	)

	return training_module

def objective(trial):
	logger.trial_number = trial.number

	early_stop = EarlyStopping(
		monitor='validation_accuracy', 
		min_delta=0.0005, 
		patience=10, 
		mode='min', 
		stopping_threshold=0.01, 
		check_on_train_epoch_end=True
	)
	
	training_module = define_trainer(trial)	
	trainer = pl.Trainer(
		default_root_dir=root_directory,
		max_epochs=150, 
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

study = optuna.create_study(
	direction='minimize',
	pruner=optuna.pruners.MedianPruner(
		n_startup_trials=10, n_warmup_steps=30, interval_steps=5
	)
)
study.optimize(objective, n_trials=100)

logger.log_best_hyperparams(
	study=study
)
