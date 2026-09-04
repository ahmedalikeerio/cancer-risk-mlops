from pathlib import Path

import joblib
import json
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import warnings
warnings.filterwarnings('ignore')

TEST_PATH = Path(
    "data/split/test.csv"
)

METRICS_PATH = Path(
    "reports/final_pipeline_metrics.json"
)

TARGET = "Risk_Level"

DROP_COLUMNS = [
    "Patient_ID",
    "Overall_Risk_Score",
]

MODEL_PATH = Path(
    "models/final_pipeline.joblib"
)

EXPERIMENT_NAME = "cancer-risk-final-model"


def main():

    print("=" * 60)
    print("FINAL PIPELINE MODEL EVALUATION")
    print("=" * 60)

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(
        columns=[TARGET] + DROP_COLUMNS
    )

    y_test = test_df[TARGET]

    # Load complete pipeline
    model = joblib.load(MODEL_PATH)

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    # High-risk metrics
    high_risk_precision = precision_score(
        y_test,
        predictions,
        labels=["High"],
        average="macro",
        zero_division=0,
    )

    high_risk_recall = recall_score(
        y_test,
        predictions,
        labels=["High"],
        average="macro",
        zero_division=0,
    )

    high_risk_f1 = f1_score(
        y_test,
        predictions,
        labels=["High"],
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "high_risk_precision": high_risk_precision,
        "high_risk_recall": high_risk_recall,
        "high_risk_f1": high_risk_f1,
    }

    print("\nFINAL PIPELINE METRICS")
    print("-" * 40)

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # Save metrics
    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_PATH,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    # MLflow evaluation run
    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="final_pipeline_evaluation"
    ):

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_artifact(
            str(METRICS_PATH)
        )

    print(
        f"\nMetrics saved: {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()