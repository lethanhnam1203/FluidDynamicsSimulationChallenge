import pandas as pd
from utils import ThaiTimeSeriesDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

SEED = 42
THAI_DF = pd.read_csv("aggregate.csv", header=None)
THAI_DF = THAI_DF.sample(frac=1, random_state=SEED).reset_index(drop=True)
DELTA: int = 10
OUTPUT_SIZE: int = 1

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
train_val_df, test_df = train_test_split(THAI_DF, test_size=TEST_RATIO, random_state=SEED)
val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
train_ratio_adjusted = TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO)
train_df, val_df = train_test_split(train_val_df, test_size=1 - train_ratio_adjusted, random_state=SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

train_dataset = ThaiTimeSeriesDataset(df=train_df, delta=DELTA, output_size=OUTPUT_SIZE)
val_dataset = ThaiTimeSeriesDataset(df=val_df, delta=DELTA, output_size=OUTPUT_SIZE)
test_dataset = ThaiTimeSeriesDataset(df=test_df, delta=DELTA, output_size=OUTPUT_SIZE)

# Create DataLoaders for each dataset
TRAIN_BATCH_SIZE: int = 32
VAL_BATCH_SIZE: int = 16
TEST_BATCH_SIZE: int = 16
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)


def main():
    pass

if __name__ == "__main__":
    main()
