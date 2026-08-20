from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_ID = "tarekmasryo/cancer-risk-factors-data"
RAW_DATA_PATH = Path("data/raw/cancer_risk.csv")


def load_and_save_dataset():
    dataset = load_dataset(DATASET_ID)

    df = dataset["train"].to_pandas()

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Dataset saved to: {RAW_DATA_PATH}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    load_and_save_dataset()