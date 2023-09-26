import pandas as pd
import os
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):

    time_col = 'Time [min]'
    voltage_col = 'Normalized Voltage [-]'
    current_col = 'Normalized Current [-]'
    temperature_col = 'Normalized Temperature [-]'
    soc_col = 'SOC [-]'

    def __init__(self, data_directory: str):
        self.data_files = []
        # for T in os.listdir(data_directory):
        for T in ['25degC']:
            T_directory = f'{data_directory}/{T}'
            file_names = filter(
                lambda f: f.endswith('parquet'), 
                os.listdir(T_directory)
            )
            self.data_files.extend([f'{T_directory}/{f}' for f in file_names])

        self.cached_data = {}

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index: int):
        if index in self.cached_data.keys():
            return self.cached_data.get(index)
        
        df = pd.read_parquet(self.data_files[index])
        t_col = self.time_col
        X_cols = [self.voltage_col, self.current_col, self.temperature_col]
        Y_col = self.soc_col
        data_length = len(df[t_col])

        t = torch.tensor(df[t_col], dtype=torch.float32).view(data_length, 1)
        X = torch.stack([torch.tensor(df[col], dtype=torch.float32) for col in X_cols], dim=1) #.view(1, len(voltage), 3)
        Y = torch.tensor(df[Y_col], dtype=torch.float32).view(data_length, 1)

        self.cached_data[index] = t, X, Y
        
        return t, X, Y


        
