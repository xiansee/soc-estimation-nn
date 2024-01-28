from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


class LSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_lstm_layers: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers
        )
        self.fc_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, X: Tensor) -> Tensor:
        lstm_output, _ = self.lstm(X)
        model_output = self.fc_layer(lstm_output)

        return model_output


class CellModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_lstm_layers: int,
    ):
        super().__init__()
        self.overpotential_model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_lstm_layers=num_lstm_layers,
        )

    def forward(self, X: Tensor) -> Tensor:
        OCV, X = X
        overpotential = self.overpotential_model.forward(X)
        overpotential = self.unnormalize_voltage(overpotential)
        cell_voltage = OCV + overpotential

        return cell_voltage

    def unnormalize_voltage(self, X: Tensor) -> Tensor:
        min_voltage = -2
        max_voltage = 0

        return (X + 1) / 2 * (max_voltage - min_voltage) + min_voltage
