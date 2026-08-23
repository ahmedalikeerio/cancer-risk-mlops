from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/cancer_risk.csv")
TRAIN_PATH = Path("data/split/train.csv")
TEST_PATH = Path("data/split/test.csv")

TARGET = "Risk_Level"


def split_data():
    df = pd.read_csv(RAW_DATA_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET],
    )

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("DATA SPLIT COMPLETED")
    print(f"Full dataset: {df.shape}")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")

    print("\nTrain distribution:")
    print(train_df[TARGET].value_counts(normalize=True))

    print("\nTest distribution:")
    print(test_df[TARGET].value_counts(normalize=True))


if __name__ == "__main__":
    split_data()