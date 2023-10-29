import sys
sys.path.append('../')
from soc_estimation_nn.helper import plot
from model_training.dataset.dataset_v1 import DatasetV1
from model_training.dataset.dataset_v2 import DatasetV2
from model_training.dataset.dataset_v3 import DatasetV3
from model_training.dataset.dataset_v4 import DatasetV4
import numpy as np
import pickle
import pandas as pd


training_dates = ['2023_10_29_T16_49_44Z', '2023_10_29_T19_22_45Z', '2023_10_29_T20_04_56Z', '2023_10_29_T19_45_23Z']
trial_nums = [36, 2, 53, 88]
datasets = [DatasetV1, DatasetV2, DatasetV3, DatasetV4]
model_pred = []
dataset_index = 5
main_dataset = DatasetV1(data_directory='../data/processed')
csv_file = main_dataset.dataset_names[dataset_index]
main_df = pd.read_parquet(f'../data/processed/25degC/{csv_file}')


for training_date, trial_num, Dataset in zip(training_dates, trial_nums, datasets):

    dataset = Dataset(data_directory='../data/processed')

    with open(f'../model_training/training_logs/{training_date}/trained_models/Trial_{trial_num}', 'rb') as binary_file:
        model = pickle.load(binary_file)

    X, Y = dataset[dataset_index]
    data_length = X.shape[0]
    input_size = X.shape[1]
    X = X.view(1, data_length, input_size)
    Y = Y.view(1, data_length, 1)
    Y_pred = model(X)

    Y = Y.view(data_length, 1).detach().numpy()
    Y_pred = Y_pred.view(data_length, 1).detach().numpy()

    rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
    model_pred.append({
        'X': dataset.data[dataset_index]['Time [min]'],
        'Y_pred': Y_pred,
        'rmse': rmse
    })


xy_data = [
    {
        'x_data': pred['X'],
        'y_data': pred['Y_pred'],
        'label': f'Predicted SOC | Architecture #{index} | RMSE = {pred["rmse"]*100:.1f} %',
        'plot_num': 3
    } for index, pred in enumerate(model_pred, 1)
]

plot(
    xy_data=[
        {
            'x_data': main_df['Time [min]'],
            'y_data': main_df['Voltage [V]'],
            'label': 'Measured Voltage',
            'plot_num': 1,
        },
        {
            'x_data': main_df['Time [min]'],
            'y_data': main_df['Current [A]'],
            'label': 'Measured Current',
            'plot_num': 2,
        },
        *xy_data,
        {
            'x_data': main_dataset.data[dataset_index]['Time [min]'],
            'y_data': main_dataset.data[dataset_index]['SOC [-]'],
            'label': 'True SOC',
            'plot_num': 3,
            'color': 'black',
            'linestyle': '--'
        }
    ],
    x_label='Time [min]',
    y_label={
        1: 'Voltage [V]',
        2: 'Current [A]',
        3: 'SOC [-]', 
    },
    title=f'LSTM Model Performance in SOC Prediction',
    fig_size=(16, 9),
)

