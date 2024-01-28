import os
from typing import Callable

import lightning as L
import pandas as pd
import torch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset, random_split


class LGHG2Dataset(Dataset):
    time_col = "Time [min]"
    voltage_col = "Voltage [V]"
    current_col = "Current [A]"
    temperature_col = "Temperature [degC]"
    capacity_col = "Capacity [Ah]"
    soc_col = "SOC [-]"
    ocv_col = "Open Circuit Voltage [V]"
    overpotential_col = "Overpotential [V]"

    norm_settings = {
        # voltage_col: (2.5, 4.2),
        current_col: (-20, 20),
        temperature_col: (-30, 50),
        soc_col: (0, 1),
        # ocv_col: (2.5, 4.2),
        overpotential_col: (-1, 1),
    }

    def __init__(self, data_directory: str):
        self.data = []
        self.dataset_names = []

        for T in ["25degC"]:
            T_directory = f"{data_directory}/{T}"
            file_names = list(
                filter(lambda f: f.endswith("parquet"), os.listdir(T_directory))
            )

            dfs = [pd.read_parquet(f"{T_directory}/{f}") for f in file_names]
            dfs = [self.calculate_overpotential(df) for df in dfs]
            dfs = [self.normalize_data(df) for df in dfs]

            self.data.extend(dfs)
            self.dataset_names.extend(file_names)
            break

    def calculate_overpotential(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.overpotential_col] = df[self.voltage_col] - df[self.ocv_col]
        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, norm_range in self.norm_settings.items():
            min_val, max_val = norm_range
            df[col] = (df[col] - min_val) / (max_val - min_val) * 2 - 1

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        df = self.data[index]

        # OCV_col = self.ocv_col
        X_cols = [self.soc_col, self.current_col, self.temperature_col]
        Y_col = self.overpotential_col  # self.voltage_col
        data_length = len(df[self.time_col])

        # OCV = torch.tensor(df[OCV_col], dtype=torch.float32).view(data_length, 1)
        X = torch.stack(
            [torch.tensor(df[col], dtype=torch.float32) for col in X_cols], dim=1
        )
        Y = torch.tensor(df[Y_col], dtype=torch.float32).view(data_length, 1)

        # return (OCV, X), Y
        return X, Y


class DataModule(L.LightningDataModule):
    """Data module that splits dataset into train, validation and test."""

    def __init__(
        self,
        data_directory: str,
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1,
    ) -> None:
        super().__init__()

        if round(sum([train_split, val_split, test_split]), 6) != 1:
            raise ValueError("All of train/val/test splits must sum up to 1.")

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.dataset = LGHG2Dataset(data_directory=data_directory)

    def setup(self, stage: str) -> None:
        num_dataset = len(self.dataset)
        num_training_set = round(self.train_split * num_dataset)
        num_validation_set = round(self.val_split * num_dataset)
        num_test_set = round(self.test_split * num_dataset)

        self.training_set, self.validation_set, self.test_set = random_split(
            self.dataset, [num_training_set, num_validation_set, num_test_set]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_set, batch_size=1, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_set, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=1)
