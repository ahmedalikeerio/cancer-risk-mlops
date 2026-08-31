from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression


TRAIN_PATH = Path("data/processed/train_processed.csv")
MODEL_PATH = Path("models/final_model.joblib")

TARGET = "Risk_Level"


def main():

    print("=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)

    # Load training data
    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    print(f"Training data: {X_train.shape}")
    print(f"Classes: {sorted(y_train.unique())}")

    # Final selected model
    model = LogisticRegression(
        C=100.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    # Train
    model.fit(X_train, y_train)

    # Save model
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nFINAL MODEL TRAINED")
    print(f"Model: {model.__class__.__name__}")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()