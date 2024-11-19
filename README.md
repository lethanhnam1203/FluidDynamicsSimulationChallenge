# FluidDynamicsSimulationChallenge
Using neural networks to solve a simulation problem in fluid dynamics


## Step 1: Data 
Done
## Step 2: Model

Implement all the time-series models that were mentioned in the research proposal for time sereis prediction. Many models can be imported from pytorch

- [x] [RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU)
- [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)
- TCN: there is no direct implementation in pytorch but from [this library](https://pypi.org/project/pytorch-tcn/) or we can implement ouselves using [Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) from pytorch
- [Full Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer) and [Transformer Encoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer)

## Step 3
Done


## Others
- Also try (shallow) machine learning methods and compare their performance to neural network methods.
- Use tensorboard and logging to record each model's performance and training history.
