from pathlib import Path
import json
import joblib

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


TRAIN_PATH = Path("data/processed/train_processed.csv")
TEST_PATH = Path("data/processed/test_processed.csv")

TARGET = "Risk_Level"
RANDOM_STATE = 42
HIGH_RISK_CLASS = "High"


def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test


def get_searches():

    return {

        "logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["lbfgs"],
            },
        ),

        "logistic_regression_balanced": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["lbfgs"],
            },
        ),

        "random_forest_balanced": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"],
            },
        ),
    }


def calculate_metrics(y_true, y_pred):

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision_weighted": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "recall_weighted": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "high_risk_precision": report.get(
            HIGH_RISK_CLASS,
            {},
        ).get("precision", 0.0),

        "high_risk_recall": report.get(
            HIGH_RISK_CLASS,
            {},
        ).get("recall", 0.0),

        "high_risk_f1": report.get(
            HIGH_RISK_CLASS,
            {},
        ).get("f1-score", 0.0),
    }

    return metrics, report


def tune_model(
    model_name,
    model,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
):

    print("\n" + "=" * 70)
    print(f"TUNING: {model_name}")
    print("=" * 70)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    with mlflow.start_run(
        run_name=f"{model_name}_tuning"
    ):

        search.fit(
            X_train,
            y_train,
        )

        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)

        metrics, report = calculate_metrics(
            y_test,
            predictions,
        )

        # Log best parameters
        mlflow.log_params(
            search.best_params_
        )

        # Log CV score
        mlflow.log_metric(
            "cv_best_macro_f1",
            search.best_score_,
        )

        # Log all evaluation metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(
                metric_name,
                value,
            )

        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "model",
        )

        # Save tuned model
        model_dir = Path("models")
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            model_dir /
            f"{model_name}_tuned.joblib"
        )

        joblib.dump(
            best_model,
            model_path,
        )

        # Save confusion matrix
        reports_dir = Path("reports")
        reports_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=sorted(y_test.unique()),
        )

        cm_path = (
            reports_dir /
            f"{model_name}_confusion_matrix.json"
        )

        with open(cm_path, "w") as f:
            json.dump(
                cm.tolist(),
                f,
                indent=4,
            )

        # Console output
        print("\nBest Parameters:")
        print(search.best_params_)

        print(
            f"\nBest CV Macro F1: "
            f"{search.best_score_:.4f}"
        )

        print("\nTest Metrics:")

        for metric_name, value in metrics.items():
            print(
                f"{metric_name}: {value:.4f}"
            )

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        print(
            f"Saved model: {model_path}"
        )

        return {
            "best_params": search.best_params_,
            "cv_best_macro_f1": search.best_score_,
            **metrics,
        }


def main():

    mlflow.set_experiment(
        "cancer-risk-hyperparameter-tuning"
    )

    X_train, X_test, y_train, y_test = load_data()

    searches = get_searches()

    results = {}

    for model_name, (
        model,
        param_grid,
    ) in searches.items():

        results[model_name] = tune_model(
            model_name=model_name,
            model=model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

    # Save results
    reports_dir = Path("reports")
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        reports_dir /
        "tuning_results.json"
    )

    with open(results_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    # Select best model based on Macro F1
    best_model_name = max(
        results,
        key=lambda name:
        results[name]["f1_macro"],
    )

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 70)

    for model_name, metrics in results.items():

        print(f"\n{model_name}")

        print(
            f"  CV Macro F1:     "
            f"{metrics['cv_best_macro_f1']:.4f}"
        )

        print(
            f"  Test Macro F1:   "
            f"{metrics['f1_macro']:.4f}"
        )

        print(
            f"  Weighted F1:     "
            f"{metrics['f1_weighted']:.4f}"
        )

        print(
            f"  High-risk Recall:"
            f" {metrics['high_risk_recall']:.4f}"
        )

        print(
            f"  Accuracy:        "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Best Params:     "
            f"{metrics['best_params']}"
        )

    print("\n" + "=" * 70)
    print("BEST TUNED MODEL")
    print("=" * 70)

    print(f"Model: {best_model_name}")

    print(
        f"Macro F1: "
        f"{results[best_model_name]['f1_macro']:.4f}"
    )

    print(
        f"High-risk Recall: "
        f"{results[best_model_name]['high_risk_recall']:.4f}"
    )

    print(
        f"\nSaved results: {results_path}"
    )


if __name__ == "__main__":
    main()