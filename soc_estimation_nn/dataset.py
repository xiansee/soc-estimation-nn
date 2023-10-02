import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader

class BatterySOCDataset(Dataset):

    time_col = 'Time [min]'
    voltage_col = 'Normalized Voltage [-]'
    current_col = 'Normalized Current [-]'
    temperature_col = 'Normalized Temperature [-]'
    soc_col = 'SOC [-]'


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


    def get_dataset_name(self, index: int):
        return self.dataset_names[index]
    

    def get_time_steps(self, index: int):
        df = self.data[index]
        data_length = len(df[self.time_col])

        t = torch.tensor(df[self.time_col], dtype=torch.float32).view(data_length, 1)
        return t


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index: int):
        df = self.data[index]

        X_cols = [self.voltage_col, self.current_col, self.temperature_col]
        Y_col = self.soc_col
        data_length = len(df[self.time_col])

        X = torch.stack([torch.tensor(df[col], dtype=torch.float32) for col in X_cols], dim=1)
        Y = torch.tensor(df[Y_col], dtype=torch.float32).view(data_length, 1)

        return X, Y
