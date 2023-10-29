from torch import nn, Tensor


class ModelV2(nn.Module):


	def __init__(
		self, 
		input_size: int, 
		hidden_size: int, 
		output_size: int, 
		num_lstm_layers: int,
		dropout_rate: float
	) -> None:
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_lstm_layers = num_lstm_layers

		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers)
		self.dropout = nn.Dropout(p=dropout_rate)
		self.fc_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
		

	def forward(
		self, 
		X: Tensor
	) -> Tensor:
		lstm_output, _ = self.lstm(X)
		dropout_output = self.dropout(lstm_output)
		model_output = self.fc_layer(dropout_output)

		return model_output
