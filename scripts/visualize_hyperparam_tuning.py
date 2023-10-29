import sys
sys.path.append('../')
from soc_estimation_nn.helper import plot
import pandas as pd
import matplotlib.pyplot as plt 

training_dates = ['2023_10_29_T16_49_44Z', '2023_10_29_T19_22_45Z', '2023_10_29_T20_04_56Z', '2023_10_29_T19_45_23Z']
architecture_num = 4

df = pd.read_csv(f'../model_training/training_logs/{training_dates[architecture_num - 1]}/HyperparametersTuning.csv')
plt.scatter(x=df['lstm_layers'], y=df['lstm_hidden_size'], c=df['validation_accuracy'], cmap='rainbow') 
plt.xlabel('Number of LSTM Layers')
plt.ylabel('Number of Neurons per Layer')
plt.title(f'Hyperparameters Tuning for LSTM Architecture #{architecture_num}')

plt.colorbar(label='Validation RMSE', orientation='vertical') 
plt.show()