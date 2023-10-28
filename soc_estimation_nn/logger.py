from datetime import datetime, timedelta
import os
import logging
import time
import sys
import psutil
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from soc_estimation_nn.helper import round_to_sig_fig
from soc_estimation_nn.metric import Metric
from torch.optim import Optimizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from optuna.trial import Trial
from optuna.study import Study
import math


class TrainingLogger(Callback):
    

    def __init__(
        self, 
        training_name: str,
        log_directory: str = os.getcwd(),
        save_log: bool = True,
        stream_log: bool = False,
    ):
        self.init_time = datetime.utcnow().strftime('%Y_%m_%d_T%H_%M_%SZ')
        self.training_name = str(training_name)
        self.log_directory = log_directory
        os.makedirs(self.log_directory, exist_ok=True)

        self.logger = logging.getLogger(training_name)
        self.logger.setLevel(logging.INFO)

        if save_log:
            self.logger.addHandler(
                logging.FileHandler(f'{self.log_directory}/{self.name}.log')
            )

        if stream_log:
            self.logger.addHandler(
                logging.StreamHandler(sys.stdout)
            )

        self.metrics_tracker = []

        self.trial_number = 0
        self.trial_start_timestamp = float('nan')
        self.training_start_timestamp = float('nan')
        self.epoch_start_timestamp = float('nan')
        self.val_acc = float('nan')

        #TODO: log every n epoch


    @property
    def name(self):
        return f'{self.init_time}-{self.training_name}'
    

    @property
    def trial_number(self) -> int:
        return self._trial_number
    

    @trial_number.setter
    def trial_number(
        self, 
        trial_number: int
    ) -> None:
        if not isinstance(trial_number, int):
            raise Exception('Trial_number needs to be of type int.')
        
        if trial_number < 0:
            raise Exception('Invalid trial_number, should be >= 0')
        
        self._trial_number = trial_number


    def log_info(
        self, 
        message: str | dict | list
    ) -> None:
        utc_timestamp = time.time()
        local_time = datetime.now()
        self.logger.info(f'{local_time} | {utc_timestamp:.6f} | {message}')


    def log_metric(
        self, 
        metric_name: str, 
        metric_data: dict
    ) -> None:
    
        file_name = f'{self.log_directory}/{metric_name}.csv'
        log_header = not os.path.isfile(file_name)                

        with open(file_name, 'a') as csv_file:

            if log_header:
                metric_header = ','.join([str(hdr) for hdr in metric_data.keys()])
                csv_file.write(f'{metric_header}\n')

            metric_values = ','.join([str(v) for v in metric_data.values()])
            csv_file.write(f'{metric_values}\n')


    def log_trial(
        self,
        trial: Trial,
        trainer_metrics: dict
    ) -> None:
        
        file_name = f'{self.log_directory}/HyperparametersTuning.csv'
        log_header = not os.path.isfile(file_name)  

        with open(file_name, 'a') as csv_file:

            if log_header:
                hyperparam_header = ','.join([
                    'trial_num', 
                    *[str(hdr) for hdr in trial.params.keys()], 
                    *[str(hdr) for hdr in trainer_metrics.keys()]
                ])
                csv_file.write(f'{hyperparam_header}\n')

            hyperparam_values = ','.join([
                str(trial.number), 
                *[str(v) for v in trial.params.values()], 
                *[str(float(v)) for v in trainer_metrics.values()]
            ])
            csv_file.write(f'{hyperparam_values}\n')


    def log_hyperparameters(
        self,
        trial_params: dict
    ) -> None:
        
        hyperparams = ', '.join([f'{param}={round_to_sig_fig(float(value), 4)}' for param, value in trial_params.items()])
        self.logger.info('')
        self.log_info(f'Trial {self.trial_number}')
        self.log_info(f'Hyperparameters for Trial {self.trial_number}: {hyperparams}')


    def log_best_hyperparams(
        self,
        study: Study,
    ) -> None:
        best_objective = study.best_value
        best_trial_num = study.best_trial.number
        best_parameters = study.best_params


        if self.trial_start_timestamp:
            hyperparam_tuning_time = str(timedelta(
                seconds=round(time.time() - self.trial_start_timestamp)
            )) 
            
        else:
            hyperparam_tuning_time = float('nan')

        self.logger.info('')
        self.log_info(f'Hyperparameters tuning completed in {hyperparam_tuning_time} (hr:min:sec).')
        best_hyperparams = ', '.join([f'{param}={round_to_sig_fig(float(value), 4)}' for param, value in best_parameters.items()])
        best_validation_accuracy = str(round_to_sig_fig(float(best_objective), 4))

        self.log_info(f'Trial {best_trial_num} has the best validation accuracy at {best_validation_accuracy}.')
        self.log_info(f'Best hyperparameters (from Trial {best_trial_num}): {best_hyperparams}')


    def track_metrics(
        self,
        metrics: Metric | list[Metric]
    ) -> None:
        
        if isinstance(metrics, Metric):
            metrics = [metrics]

        for metric in metrics:

            if metric in self.metrics_tracker:
                continue

            #TODO: verify metric is trackable
            
            self.metrics_tracker.append(metric)
            self.log_info(f'Tracking metric: {metric.name}')
        

    def compute_and_log_metrics(callback_func):

        def wrapped_callback(*args):
            callback_func(*args)
            self = args[0]

            for metric in self.metrics_tracker:

                compute_fn = getattr(metric, f'compute_{callback_func.__name__}', None)

                if compute_fn:
                    metric_values = compute_fn(*args[1:])

                    if metric_values:
                        pl_module = args[2]
                        epoch_num, global_step = pl_module.current_epoch, pl_module.global_step

                        self.log_metric(
                            metric_name=metric.name,
                            metric_data={
                                'trial_num': self.trial_number,
                                'epoch_num': epoch_num,
                                'global_step': global_step,
                                **metric_values
                            }
                        )

        return wrapped_callback


    @compute_and_log_metrics
    def on_train_start(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ):
        self.log_info(f'Training started for Trial {self.trial_number}.')
        self.trial_start_timestamp = time.time() if math.isnan(self.trial_start_timestamp) else self.trial_start_timestamp
        self.training_start_timestamp = time.time()


    @compute_and_log_metrics
    def on_fit_end(
        self,
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ):
        if self.training_start_timestamp:
            training_time = str(timedelta(
                seconds=round(time.time() - self.training_start_timestamp)
            )) 
            
        else:
            training_time = float('nan')

        self.log_info(f'Fit completed for Trial {self.trial_number} in {training_time} (hr:min:sec).')
    

    @compute_and_log_metrics
    def on_before_zero_grad(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
        optimizer: Optimizer
    ) -> None:
        return


    @compute_and_log_metrics
    def on_train_epoch_start(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        self.epoch_start_timestamp = time.time()


    @compute_and_log_metrics
    def on_train_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        return


    @compute_and_log_metrics
    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        epoch_num, global_step = pl_module.current_epoch, pl_module.global_step
        
        if global_step == 0:
            return

        training_loss = round_to_sig_fig(float(trainer.logged_metrics.get('training_loss', 'nan')), 4)
        validation_accuracy = round_to_sig_fig(float(trainer.logged_metrics.get('validation_accuracy', 'nan')), 4)
        time_per_epoch = round_to_sig_fig((time.time() - self.epoch_start_timestamp) * 1000, 4) if self.epoch_start_timestamp else float('nan')
        cpu_usage = psutil.cpu_percent() # %
        mem_usage = round_to_sig_fig(psutil.virtual_memory().used / 10 ** 9, 4) # GB

        self.log_info((
            f'trial_number={self.trial_number}, '
            f'epoch={epoch_num}, '
            f'global_step={global_step}, '
            f'training_loss={training_loss}, '
            f'validation_accuracy={validation_accuracy}, '
            f'time_per_epoch={time_per_epoch}ms, '
            f'cpu_usage={cpu_usage}%, '
            f'memory_usage={mem_usage}GB'
        ))
        self.log_metric(
            metric_name='TrainingAndValidationLoss',
            metric_data={
                'trial_number': self.trial_number,
                'epoch_num': epoch_num,
                'global_step': global_step,
                'training_loss': training_loss,
                'validation_accuracy': validation_accuracy,
            }
        )
        self.val_acc = validation_accuracy


    @compute_and_log_metrics
    def on_validation_batch_end(self, 
		trainer: pl.Trainer, 
		pl_module: pl.LightningModule, 
		outputs: STEP_OUTPUT | None, 
		batch: Any, 
		batch_idx: int, 
		dataloader_idx: int = 0
	) -> None:
        return


    @compute_and_log_metrics
    def on_train_batch_end(self, 
		trainer: pl.Trainer, 
		pl_module: pl.LightningModule, 
		outputs: STEP_OUTPUT | None, 
		batch: Any, 
		batch_idx: int, 
		dataloader_idx: int = 0
	) -> None:
        return
    