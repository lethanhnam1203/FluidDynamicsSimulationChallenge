import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
import pandas as pd


class ThaiTimeSeriesDataset(Dataset):
    """Thai Time Series Dataset for Fluid Dynamics Simulation."""
    def __init__(self, df, delta: int=5, output_size: int=1):
        """
        Args:
            df (pd.DataFrame): The input dataframe, where each row corresponds to a time series.
            delta (int): Number of previous time steps to use for prediction (delta).
            output_size (int): Number of time steps to predict (output size).
        """
        self.initial_params = df.iloc[:, :5].values  # First 5 columns as initial parameters
        self.time_series = df.iloc[:, 5:].values     # Next 6000 columns as time series
        self.delta = delta
        self.output_size = output_size

    def __len__(self) -> int:
        # Total samples = number of rows * (6000 - delta - output_size + 1)
        return self.time_series.shape[0] * (self.time_series.shape[1] - self.delta - self.output_size + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate the row and time step corresponding to idx
        series_idx: int = idx // (self.time_series.shape[1] - self.delta - self.output_size + 1)
        time_step: int = idx % (self.time_series.shape[1] - self.delta - self.output_size + 1)

        # Get initial parameters for the time series
        initial_params: np.ndarray = self.initial_params[series_idx]
        
        # Get the previous delta time steps
        previous_steps: np.ndarray = self.time_series[series_idx, time_step:time_step + self.delta]
        
        # Get the target for the next time steps (output_size)
        target: np.ndarray = self.time_series[series_idx, time_step + self.delta:time_step + self.delta + self.output_size]
        
        # Return a tuple of (initial_params, previous_steps, target)
        return (
            torch.tensor(initial_params, dtype=torch.float32),
            torch.tensor(previous_steps, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )
