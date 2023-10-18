from datetime import datetime
import os
import logging
import time
import sys
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from soc_estimation_nn.helper import round_to_sig_fig
from soc_estimation_nn.metric import Metric
from torch.optim import Optimizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


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


    @property
    def name(self):
        return f'{self.init_time}-{self.training_name}'


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


    def track_metric(
        self,
        metric: Metric
    ) -> None:
        
        if metric not in self.metrics_tracker:
            self.metrics_tracker.append(metric)

        self.log_info(f'Tracking metric: {metric.name}')


    def on_train_start(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ):
        self.log_info('Training started.')


    def on_fit_end(
        self,
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ):
        self.log_info('Fit completed.')
      

    def on_before_zero_grad(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
        optimizer: Optimizer
    ) -> None:
        epoch_num, global_step = pl_module.current_epoch, pl_module.global_step

        for metric in self.metrics_tracker:

            compute_fn = getattr(metric, 'compute_on_before_zero_grad', None)

            if compute_fn:
                metric_values = compute_fn(
                    trainer=trainer, 
                    pl_module=pl_module, 
                    optimizer=optimizer
                )

                if metric_values:
                    self.log_metric(
                        metric_name=metric.name,
                        metric_data={
                            'epoch_num': epoch_num,
                            'global_step': global_step,
                            **metric_values
                        }
                    )


    def on_train_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        
        epoch_num, global_step = pl_module.current_epoch, pl_module.global_step

        for metric in self.metrics_tracker:

            compute_fn = getattr(metric, 'compute_on_train_epoch_end', None)

            if compute_fn:
                metric_values = compute_fn(
                    trainer=trainer, 
                    pl_module=pl_module, 
                )

                if metric_values:
                    self.log_metric(
                        metric_name=metric.name,
                        metric_data={
                            'epoch_num': epoch_num,
                            'global_step': global_step,
                            **metric_values
                        }
                    )


    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        
        epoch_num, global_step = pl_module.current_epoch, pl_module.global_step

        if global_step == 0:
            return

        training_loss = float(trainer.logged_metrics.get('training_loss', 'nan'))
        validation_accuracy = float(trainer.logged_metrics.get('validation_accuracy', 'nan'))

        self.log_info((
            f'epoch={epoch_num}, '
            f'global_step={global_step}, '
            f'training_loss={round_to_sig_fig(training_loss, 6)}, '
            f'validation_accuracy={round_to_sig_fig(validation_accuracy, 6)}'
        ))
        self.log_metric(
            metric_name='TrainingAndValidationLoss',
            metric_data={
                'epoch_num': epoch_num,
                'global_step': global_step,
                'training_loss': training_loss,
                'validation_accuracy': validation_accuracy,
            }
        )

        for metric in self.metrics_tracker:

            compute_fn = getattr(metric, 'compute_on_validation_epoch_end', None)

            if compute_fn:
                metric_values = compute_fn(
                    trainer=trainer, 
                    pl_module=pl_module, 
                )

                if metric_values:
                    self.log_metric(
                        metric_name=metric.name,
                        metric_data={
                            'epoch_num': epoch_num,
                            'global_step': global_step,
                            **metric_values
                        }
                    )


    def on_validation_batch_end(self, 
		trainer: pl.Trainer, 
		pl_module: pl.LightningModule, 
		outputs: STEP_OUTPUT | None, 
		batch: Any, 
		batch_idx: int, 
		dataloader_idx: int = 0
	) -> None:
        epoch_num, global_step = pl_module.current_epoch, pl_module.global_step
        
        for metric in self.metrics_tracker:

            compute_fn = getattr(metric, 'compute_on_validation_batch_end', None)

            if compute_fn:
                metric_values = compute_fn(
                    trainer=trainer, 
                    pl_module=pl_module,
                    outputs=outputs, 
		            batch=batch, 
		            batch_idx=batch_idx, 
		            dataloader_idx=dataloader_idx 
                )

                if metric_values:
                    self.log_metric(
                        metric_name=metric.name,
                        metric_data={
                            'epoch_num': epoch_num,
                            'global_step': global_step,
                            **metric_values
                        }
                    )


    def on_train_batch_end(self, 
		trainer: pl.Trainer, 
		pl_module: pl.LightningModule, 
		outputs: STEP_OUTPUT | None, 
		batch: Any, 
		batch_idx: int, 
		dataloader_idx: int = 0
	) -> None:
        epoch_num, global_step = pl_module.current_epoch, pl_module.global_step
        
        for metric in self.metrics_tracker:

            compute_fn = getattr(metric, 'compute_on_train_batch_end', None)

            if compute_fn:
                metric_values = compute_fn(
                    trainer=trainer, 
                    pl_module=pl_module,
                    outputs=outputs, 
		            batch=batch, 
		            batch_idx=batch_idx, 
		            dataloader_idx=dataloader_idx 
                )

                if metric_values:
                    self.log_metric(
                        metric_name=metric.name,
                        metric_data={
                            'epoch_num': epoch_num,
                            'global_step': global_step,
                            **metric_values
                        }
                    )
    