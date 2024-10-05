import pandas as pd
from utils import ThaiTimeSeriesDataset

RANDOM_SEED: int = 42
THAI_DF = pd.read_csv("aggregate.csv", header=None)
THAI_DF = THAI_DF.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True) 
DELTA: int = 10
OUTPUT_SIZE: int = 1
TRAIN_RATIO: float = 0.80
VAL_RATIO: float= 0.10
TEST_RATIO: float = 1.0 - TRAIN_RATIO - VAL_RATIO

full_dataset = ThaiTimeSeriesDataset(df=THAI_DF, delta=DELTA, output_size=OUTPUT_SIZE)
print(len(full_dataset))
print(full_dataset[0])

def main():
    pass

if __name__ == "__main__":
    main()