import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor
import lightning.pytorch as pl

class BatterySOCDataset(Dataset):

    time_col = 'Time [min]'
    voltage_col = 'Voltage [V]'
    current_col = 'Current [A]'
    temperature_col = 'Temperature [degC]'
    capacity_col = 'Capacity [Ah]'
    soc_col = 'SOC [-]'

    norm_settings = {
        voltage_col: (2.5, 4.2),
        current_col: (-20, 20),
        temperature_col: (-30, 50),
    }

    def __init__(self, data_directory: str):
        self.data = []
        self.dataset_names = []
        
        for T in ['25degC']: # temporarily reducing training data to just one temperature // for T in os.listdir(data_directory):
            T_directory = f'{data_directory}/{T}'
            file_names = list(filter(
                lambda f: f.endswith('parquet'), 
                os.listdir(T_directory)
            ))
            self.data.extend([pd.read_parquet(f'{T_directory}/{f}') for f in file_names])
            self.dataset_names.extend(file_names)
            break


    def get_dataset_name(self, index: int):
        return self.dataset_names[index]
    

    def get_time_steps(self, index: int):
        df = self.data[index]
        data_length = len(df[self.time_col])

        t = torch.tensor(df[self.time_col], dtype=torch.float32).view(data_length, 1)
        return t
    

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, norm_range in self.norm_settings.items():
            min_val, max_val = norm_range
            df[col] = (df[col] - min_val) / (max_val - min_val) * 2 - 1

        return df


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index: int):
        df = self.data[index]
        df = self.normalize_data(df)

        X_cols = [self.voltage_col, self.current_col, self.temperature_col]
        Y_col = self.soc_col
        data_length = len(df[self.time_col])

        X = torch.stack([torch.tensor(df[col], dtype=torch.float32) for col in X_cols], dim=1)
        Y = torch.tensor(df[Y_col], dtype=torch.float32).view(data_length, 1)

        return X, Y


class BatterySOCDataModule(pl.LightningDataModule):


    def __init__(
            self, 
            data_directory: str, 
            train_split: float = 0.7, 
            val_split: float = 0.2,
            test_split: float = 0.1
        ):
        super().__init__()
        self.data_directory = data_directory

        if round(sum([train_split, val_split, test_split]), 6) != 1:
            raise ValueError('All of train/val/test splits must sum up to 1.')
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split


    def setup(self, stage: str):
        dataset = BatterySOCDataset(self.data_directory)
        num_dataset = len(dataset)
        num_training_set = round(self.train_split * num_dataset)
        num_validation_set = round(self.val_split * num_dataset)
        num_test_set = round(self.test_split * num_dataset)

        self.training_set, self.validation_set, self.test_set = random_split(
            dataset, [num_training_set, num_validation_set, num_test_set]
        )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_set, batch_size=1, shuffle=True)
    

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_set, batch_size=1, shuffle=True)
    

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=1, shuffle=True)