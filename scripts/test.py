import sys
sys.path.append('../')
import pandas as pd
from soc_estimation_nn.helper import plot
from model_training.model.model_v1 import ModelV1
import torch
from model_training.dataset.dataset_v2 import DatasetV2
from model_training.dataset.dataset_v1 import DatasetV1
import numpy as np

# dataset = DatasetV2(data_directory='../data/processed')

df = pd.read_parquet('../data/processed/25degC/551_LA92.parquet')
print(df['Timestamp'].iloc[0])
print(df['Timestamp'].iloc[1])
df['Current [A]'] = df['Capacity [Ah]'].diff() / (df['Time [min]'].diff() / 60) 
print(df)

# model = ModelV1(
#     input_size=3, 
#     hidden_size=50,
#     output_size=1, 
#     num_lstm_layers=1
# )
# checkpoint = torch.load('../model_training/training_logs/2023_10_28_T21_45_31Z/checkpoints/epoch=149-step=1350.ckpt')
# new_chkpt = {}
# for k, v in checkpoint['state_dict'].items():
#     new_k = k.replace('model.', '')
#     new_chkpt[new_k] = v

# model.load_state_dict(new_chkpt)

# X, Y = dataset[10]
# data_length = X.shape[0]
# X = X.view(1, data_length, 3)
# Y = Y.view(1, data_length, 1)
# Y_pred = model(X)

# Y = Y.view(data_length, 1).detach().numpy()
# Y_pred = Y_pred.view(data_length, 1).detach().numpy()

# rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))

# plot(
#     xy_data=[
#         {
#             'x_data': list(range(0, data_length)),
#             'y_data': Y,
#             'label': 'Measured',
#             'plot_num': 1
#         },
#         {
#             'x_data': list(range(0, data_length)),
#             'y_data': Y_pred,
#             'label': 'Predicted',
#             'plot_num': 1
#         },
#     ],
#     x_label='Time [min]',
#     y_label={
#         1: 'SOC [-]', 
#     },
#     title=f'Model Performance (RMSE = {rmse*100:.3f}%)',
#     fig_size=(10, 12.5),
# )

