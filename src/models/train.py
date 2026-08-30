from pathlib import Path
import json
import joblib

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.evaluate import evaluate_model


TRAIN_PATH = Path("data/processed/train_processed.csv")
TEST_PATH = Path("data/processed/test_processed.csv")

TARGET = "Risk_Level"
RANDOM_STATE = 42


def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test


def get_models():

    return {

        "logistic_regression": {
            "model": LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "none",
            },
        },

        "logistic_regression_balanced": {
            "model": LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "params": {
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "balanced",
            },
        },

        "random_forest_balanced": {
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": "None",
                "min_samples_split": 2,
                "class_weight": "balanced",
            },
        },
    }


def train_model(
    model_name,
    model_config,
    X_train,
    X_test,
    y_train,
    y_test,
):

    model = model_config["model"]
    params = model_config["params"]

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_param("model", model_name)

        for name, value in params.items():
            mlflow.log_param(name, value)

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        metrics = evaluate_model(
            y_true=y_test,
            y_pred=predictions,
        )

        # Save model
        models_dir = Path("models")
        models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            models_dir / f"{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
        )

        print(
            f"Saved model: {model_path}"
        )

        return metrics


def main():

    mlflow.set_experiment(
        "cancer-risk-classification"
    )

    X_train, X_test, y_train, y_test = load_data()

    models = get_models()

    results = {}

    for model_name, model_config in models.items():

        metrics = train_model(
            model_name=model_name,
            model_config=model_config,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results[model_name] = metrics

    # Save all metrics for DVC
    reports_dir = Path("reports")
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        reports_dir / "metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    # Find best model using Macro F1
    best_model_name = max(
        results,
        key=lambda name: results[name]["f1_macro"],
    )

    print("\n")
    print("=" * 70)
    print("MODEL EXPERIMENT SUMMARY")
    print("=" * 70)

    for model_name, metrics in results.items():

        print(f"\n{model_name}")

        print(
            f"  Accuracy:    "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Macro F1:    "
            f"{metrics['f1_macro']:.4f}"
        )

        print(
            f"  Weighted F1: "
            f"{metrics['f1_weighted']:.4f}"
        )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(f"Model: {best_model_name}")

    print(
        f"Macro F1: "
        f"{results[best_model_name]['f1_macro']:.4f}"
    )

    print(
        f"\nSaved metrics: {metrics_path}"
    )


if __name__ == "__main__":
    main()