from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression


TRAIN_PATH = Path("data/processed/train_processed.csv")
MODEL_PATH = Path("models/final_model.joblib")

TARGET = "Risk_Level"

MODEL_NAME = "CancerRiskModel"


def main():

    print("=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    print(f"Training data: {X_train.shape}")
    print(f"Classes: {sorted(y_train.unique())}")

    model = LogisticRegression(
        C=100.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    mlflow.set_experiment(
        "cancer-risk-final-model"
    )

    with mlflow.start_run(
        run_name="final_logistic_regression"
    ):

        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "C": 100.0,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 2000,
        })

        # Log training information
        mlflow.log_params({
            "training_rows": len(X_train),
            "features": X_train.shape[1],
        })

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # Save local copy
        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            MODEL_PATH,
        )

        print("\nFINAL MODEL TRAINED")
        print(f"Local model: {MODEL_PATH}")
        print(f"MLflow model: {MODEL_NAME}")

    print("\nModel registered successfully.")


if __name__ == "__main__":
    main()