import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.nn import RNN, LSTM, GRU, MSELoss, Transformer


class ThaiRNN(nn.Module):
    def __init__(self, initial_values_size: int=5, hidden_size: int=64, num_layers: int=1, input_size:int=1, output_size: int=1):
        """
        Args:
            initial_values_size (int): Number of initial values.
            hidden_size (int): Number of hidden units in the RNN.
            num_layers (int): Number of RNN layers.
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        super(ThaiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = RNN(input_size + initial_values_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, initial_values: torch.Tensor, prev_time_steps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            initial_values (torch.Tensor): Shape (batch_size, initial_values_size).
            prev_time_steps (torch.Tensor): Shape (batch_size, delta).
        
        Returns:
            torch.Tensor: Predicted next time step(s), shape (batch_size, output_size).
        """
        if len(prev_time_steps.shape) == 2:
            prev_time_steps = prev_time_steps.unsqueeze(-1)
        initial_values = initial_values.unsqueeze(1).repeat(1, prev_time_steps.size(1), 1)
        assert initial_values.size(0) == prev_time_steps.size(0) # batch size
        assert initial_values.size(1) == prev_time_steps.size(1) # sequence length
        assert initial_values.size(2) + prev_time_steps.size(2) == self.rnn.input_size
        rnn_input = torch.cat([initial_values, prev_time_steps], dim=-1)
        h0 = torch.zeros(self.num_layers, rnn_input.size(0), self.hidden_size).to(prev_time_steps.device)
        out, _ = self.rnn(rnn_input, h0)
        out = self.fc(out[:, -1, :])
        return out
    

class NamRNN(nn.Module):
    def __init__(self, initial_values_size: int=5, hidden_size: int=64, num_layers: int=1, input_size:int=1, output_size: int=1):
        """
        Args:
            initial_values_size (int): Number of initial values.
            hidden_size (int): Number of hidden units in the RNN.
            num_layers (int): Number of RNN layers.
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        super(NamRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        # RNN layer expects input_size + initial_values_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Linear layer for output
        self.fc = nn.Linear(hidden_size, output_size)

        # Linear layer to transform initial values to hidden state (h0)
        self.initial_values_to_hidden = nn.Linear(initial_values_size, hidden_size)


    def forward(self, initial_values: torch.Tensor, prev_time_steps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            initial_values (torch.Tensor): Shape (batch_size, initial_values_size).
            prev_time_steps (torch.Tensor): Shape (batch_size, delta).
        
        Returns:
            torch.Tensor: Predicted next time step(s), shape (batch_size, output_size).
        """
        if len(prev_time_steps.shape) == 2:
            prev_time_steps = prev_time_steps.unsqueeze(-1)  # Make sure prev_time_steps is (batch_size, seq_len, 1)

        assert initial_values.size(0) == prev_time_steps.size(0)  # batch size
        assert initial_values.size(1) == self.initial_values_to_hidden.in_features

        rnn_input = prev_time_steps # (batch_size, seq_len, input_size=1)

        # Initialize hidden state from initial values
        h0 = self.initial_values_to_hidden(initial_values) # (batch_size, hidden_size)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1) # (num_layers, batch_size, hidden_size)

        # Pass the input through the RNN
        out, _ = self.rnn(rnn_input, h0) # (batch_size, seq_len, hidden_size) 

        # Output prediction (last time step)
        out = self.fc(out[:, -1, :]) # (batch_size, output_size)

        return out
