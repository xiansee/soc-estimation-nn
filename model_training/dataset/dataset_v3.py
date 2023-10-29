import pandas as pd
import os
import torch
from torch.utils.data import Dataset


class DatasetV3(Dataset):

    timestamp_col = 'Timestamp'
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

            self.data.extend([
                self.normalize_data(self.downsample_data(pd.read_parquet(f'{T_directory}/{f}'))) \
                for f in file_names
            ])
            self.dataset_names.extend(file_names)
            break    


    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, norm_range in self.norm_settings.items():
            min_val, max_val = norm_range
            df[col] = (df[col] - min_val) / (max_val - min_val) * 2 - 1

        return df
    
    
    def use_average_current(self, df:pd.DataFrame) -> pd.DataFrame:        
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df = df.set_index(self.timestamp_col)
        df = df.resample('1S').mean()
        df = df.dropna()

        return df


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