import pandas as pd
from utils import ThaiTimeSeriesDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
from typing import Tuple, List, Iterable, Dict, Union
from models import ThaiRNN, NamRNN

SEED: int = 42
THAI_DF = pd.read_csv("aggregate.csv", header=None)
THAI_DF = THAI_DF.sample(frac=1, random_state=SEED).reset_index(drop=True)
DELTA: int = 10
OUTPUT_SIZE: int = 1
TRAIN_BATCH_SIZE: int = 256
VAL_BATCH_SIZE: int = TRAIN_BATCH_SIZE // 2
TEST_BATCH_SIZE: int = VAL_BATCH_SIZE
EPOCHS: int = 1
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
train_ratio_adjusted = TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO)

NUM_EPOCHS = 10

def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0 # accumulate loss over all batches in an epoch
    n_total_steps = len(train_loader)
    for batch_idx, (initial_values, prev_time_steps, target) in enumerate(train_loader):
        initial_values, prev_time_steps, target = initial_values.to(DEVICE), prev_time_steps.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(initial_values, prev_time_steps)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx+1) % 1000 == 0:
            logging.info(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{n_total_steps}], Train loss: {loss.item():.3f}"
            )
    average_loss = running_loss / n_total_steps
    logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average train loss: {average_loss:.3f}")
    return average_loss

def validate(
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    epoch: int,
) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (initial_values, prev_time_steps, target) in enumerate(val_loader):
            initial_values, prev_time_steps, target = initial_values.to(DEVICE), prev_time_steps.to(DEVICE), target.to(DEVICE)
            output = model(initial_values, prev_time_steps)
            loss = criterion(output, target)
            running_loss += loss.item()
            if (batch_idx+1) % 100 == 0:
                logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(val_loader)}], Val loss: {loss.item():.3f}")
    average_loss = running_loss / len(val_loader)
    logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average val loss: {average_loss:.3f}")
    return average_loss


def test(
    test_loader: DataLoader,
    model: nn.Module,
    metric_func: nn.Module,
) -> float:
    model.eval()
    metric_score = 0.0
    with torch.no_grad():
        for batch_idx, (initial_values, prev_time_steps, target) in enumerate(test_loader):
            initial_values, prev_time_steps, target = initial_values.to(DEVICE), prev_time_steps.to(DEVICE), target.to(DEVICE)
            output = model(initial_values, prev_time_steps)
            metric_score += metric_func(output, target).item()
            if (batch_idx+1) % 100 == 0:
                logging.info(f"Step [{batch_idx+1}/{len(test_loader)}], Test metric: {metric_score:.3f}")
    average_metric = metric_score / len(test_loader)
    logging.info(f"Average test metric: {average_metric:.3f}")
    

def get_model(model_name: str, **kwargs) -> nn.Module:
    if model_name == "thai_rnn":
        return ThaiRNN(**kwargs)
    elif model_name == "nam_rnn":
        return NamRNN(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported yet!")


def main():
    train_val_df, test_df = train_test_split(THAI_DF, test_size=TEST_RATIO, random_state=SEED)
    train_df, val_df = train_test_split(train_val_df, test_size=1 - train_ratio_adjusted, random_state=SEED)

    train_dataset = ThaiTimeSeriesDataset(df=train_df, delta=DELTA, output_size=OUTPUT_SIZE)
    val_dataset = ThaiTimeSeriesDataset(df=val_df, delta=DELTA, output_size=OUTPUT_SIZE)
    test_dataset = ThaiTimeSeriesDataset(df=test_df, delta=DELTA, output_size=OUTPUT_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    parser = argparse.ArgumentParser(description="Train and test a neural network on the Thai Fluid Dynamics Simulation dataset.")
    parser.add_argument("--model", type=str, default="thai_rnn", help="Model to use for training.")
    args = parser.parse_args()
    model = get_model(args.model)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss(reduction="mean")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    lowest_val_loss = float("inf")
    best_epoch = 0
    logging.basicConfig = logging.basicConfig(
        level=logging.INFO,
        filename=f"logs/log_{args.model}.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"{'-' * 20} Starting training {'-' * 20}")

    for epoch in range(NUM_EPOCHS):
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, model, criterion, epoch)
        lr_scheduler.step()
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"models/{args.model}.pth")
    logging.info(f"Best epoch: {best_epoch+1}, Lowest val loss: {lowest_val_loss:.3f}")
    logging.info(f"{'-' * 20} Finished training, now Testing {'-' * 20}")
    model.load_state_dict(torch.load(f"models/{args.model}.pth"))
    test(test_loader, model, criterion)


if __name__ == "__main__":
    main()
