import torch.nn as nn
import torch


class SOCEstNN(nn.Module):
		
	def __init__(
		self, 
		input_size: int, 
		hidden_size: int, 
		output_size: int, 
		num_lstm_layers: int
	) -> None:
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_lstm_layers = num_lstm_layers

		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers)
		self.dropout = nn.Dropout(p=0.5)
		self.fc_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
		self.output_layer = nn.Sigmoid()


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		lstm_output, _ = self.lstm(X)
		dropout_output = self.dropout(lstm_output)
		fc_layer_output = self.fc_layer(dropout_output)
		model_output = self.output_layer(fc_layer_output)

		return model_output


class RMSELoss(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		mse_loss_fn = nn.MSELoss()
		rmse_loss = torch.sqrt(mse_loss_fn(x, y))
		return rmse_loss