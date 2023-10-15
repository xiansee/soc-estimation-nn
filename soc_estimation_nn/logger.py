from datetime import datetime
import os
import logging
import time
import sys
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
        self.logger.info(f'{local_time} | {utc_timestamp} | {message}')


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


    def track(
        self,
        metric: Metric
    ) -> None:
        remove_duplicates = lambda elem: list(set(elem))

        self.metrics_tracker.append(metric)
        self.metrics_tracker = remove_duplicates(self.metrics_tracker)
        

    def on_before_zero_grad(
        self, 
        trainer: pl.Trainer, 
        training_module: pl.LightningModule,
        optimizer: Optimizer
    ) -> None:

        for metric in filter(
            lambda m: m.compute_on_before_zero_grad, 
            self.metrics_tracker
        ):
            metric_values = metric.compute(trainer, training_module)

            if metric_values:
                epoch_num, global_step = training_module.current_epoch, training_module.global_step
                metric_data = {
                    'epoch_num': epoch_num,
                    'global_step': global_step,
                    **metric_values
                }

                self.log_metric(
                    metric_name=metric.name,
                    metric_data=metric_data,
                )


    def on_train_epoch_end(
        self, 
        trainer: pl.Trainer, 
        training_module: pl.LightningModule
    ) -> None:
        
        for metric in filter(
            lambda m: m.compute_on_train_epoch_end, 
            self.metrics_tracker
        ):
            metric_values = metric.compute(trainer, training_module)

            if metric_values:
                epoch_num, global_step = training_module.current_epoch, training_module.global_step
                metric_data = {
                    'epoch_num': epoch_num,
                    'global_step': global_step,
                    **metric_values
                }

                self.log_metric(
                    metric_name=metric.name,
                    metric_data=metric_data,
                )


    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        training_module: pl.LightningModule
    ) -> None:
        
        for metric in filter(
            lambda m: m.compute_on_validation_epoch_end, 
            self.metrics_tracker
        ):
            metric_values = metric.compute(trainer, training_module)

            if metric_values:
                epoch_num, global_step = training_module.current_epoch, training_module.global_step
                metric_data = {
                    'epoch_num': epoch_num,
                    'global_step': global_step,
                    **metric_values
                }

                self.log_metric(
                    metric_name=metric.name,
                    metric_data=metric_data,
                )

    