# FluidDynamicsSimulationChallenge
Using neural networks to solve a simulation problem in fluid dynamics


## Step 1: Data 

1.1. Implement a dataset object based on the `torch.utils.data.Dataset` library.

Documentation can be found [here](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

It is required to implement two methods of `__getitem__()` and `__len__()`. The former is to return a  data sample (both input X and label y) for a given index key. The latter is to return the size of the dataset (how many data samples?).

Based on the dataset object, create three instances: `train_dataset, val_dataset, `test_dataset`. The split ratio can 85:5:10. 

Note that for data augmentation purposes, we may need to apply `transform` for the train and validation datasets.

1.2. Implement data loaders

Documentation can be found [here](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

Based on the three datasets `train_dataset, val_dataset, `test_dataset`, we implement the corresponding `train_loader, val_loader, `test_loader`

## Step 2: Model

Implement all the time-series models that were mentioned in the research proposal for time sereis prediction. Many models can be imported from pytorch

- [RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU)
- [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)
- TCN: there is no direct implementation in pytorch but from [this library](https://pypi.org/project/pytorch-tcn/) or we can implement ouselves using [Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) from pytorch
- [Full Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer) and [Transformer Encoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer)

## Step 3
- Write a training loop. Please follow the `main` function of [this code](https://github.com/lethanhnam1203/CarChallenge/blob/main/main.py)



## Others
- Also try (shallow) machine learning methods and compare their performance to neural network methods.
- Use tensorboard and logging to record each model's performance and training history.


