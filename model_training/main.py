import yaml
from data.data_module import DataModule
from model.cell_model import LSTM, CellModel

from model_trainer.core.hyperparameter_tuning import run_hyperparameter_tuning
from model_trainer.core.training_config import process_user_config

if __name__ == "__main__":
    with open("config.yaml", "r") as confg_file:
        user_config = yaml.safe_load(confg_file)

    user_config["data_module"].update({"data_module": DataModule})
    user_config["model"].update({"model": LSTM})

    training_config = process_user_config(user_config)
    study = run_hyperparameter_tuning(training_config=training_config)
