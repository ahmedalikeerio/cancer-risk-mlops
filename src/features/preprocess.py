from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


TRAIN_PATH = Path("data/split/train.csv")
TEST_PATH = Path("data/split/test.csv")

TRAIN_OUTPUT = Path("data/processed/train_processed.csv")
TEST_OUTPUT = Path("data/processed/test_processed.csv")
PREPROCESSOR_PATH = Path("data/processed/preprocessor.joblib")

TARGET = "Risk_Level"

DROP_COLUMNS = [
    "Patient_ID",
    "Overall_Risk_Score",
]


def preprocess_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET] + DROP_COLUMNS)
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET] + DROP_COLUMNS)
    y_test = test_df[TARGET]

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                "passthrough",
                numerical_columns,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
        ]
    )

    # IMPORTANT: fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Test data is only transformed
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
    )

    test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
    )

    train_processed[TARGET] = y_train.values
    test_processed[TARGET] = y_test.values

    TRAIN_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_processed.to_csv(TRAIN_OUTPUT, index=False)
    test_processed.to_csv(TEST_OUTPUT, index=False)

    dump(preprocessor, PREPROCESSOR_PATH)

    print("PREPROCESSING COMPLETED")
    print(f"Train input: {train_df.shape}")
    print(f"Test input: {test_df.shape}")
    print(f"Train processed: {train_processed.shape}")
    print(f"Test processed: {test_processed.shape}")
    print(f"Saved preprocessor: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    preprocess_data()