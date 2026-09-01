from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


TEST_PATH = Path("data/processed/test_processed.csv")
MODEL_PATH = Path("models/final_model.joblib")
METRICS_PATH = Path("reports/final_model_metrics.json")

TARGET = "Risk_Level"

MLFLOW_EXPERIMENT = "cancer-risk-final-model"


def main():

    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "precision_weighted": float(
            precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "high_risk_precision": float(
            report["High"]["precision"]
        ),
        "high_risk_recall": float(
            report["High"]["recall"]
        ),
        "high_risk_f1": float(
            report["High"]["f1-score"]
        ),
    }

    print("\n===== FINAL MODEL RESULTS =====")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=["High", "Low", "Medium"],
        )
    )

    # Save local metrics
    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(METRICS_PATH, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    # Log metrics to MLflow
    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    with mlflow.start_run(
        run_name="final_model_evaluation"
    ):

        mlflow.log_metrics(metrics)

        mlflow.set_tag(
            "model_name",
            "CancerRiskModel",
        )

        mlflow.set_tag(
            "evaluation_type",
            "final_test_evaluation",
        )

    print(
        f"\nSaved metrics: {METRICS_PATH}"
    )

    print("Metrics logged to MLflow.")


if __name__ == "__main__":
    main()