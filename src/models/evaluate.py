from pathlib import Path

import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(y_true, y_pred):

    labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        zero_division=0,
    )

    print("\n===== EVALUATION RESULTS =====")

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision Weighted: {precision_weighted:.4f}")
    print(f"Recall Weighted:    {recall_weighted:.4f}")
    print(f"F1 Weighted:        {f1_weighted:.4f}")
    print(f"F1 Macro:           {f1_macro:.4f}")

    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Cancer Risk Classification - Confusion Matrix"
    )

    plt.tight_layout()

    confusion_matrix_path = (
        REPORT_DIR / "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # MLflow metrics
    mlflow.log_metric(
        "accuracy",
        accuracy,
    )

    mlflow.log_metric(
        "precision_weighted",
        precision_weighted,
    )

    mlflow.log_metric(
        "recall_weighted",
        recall_weighted,
    )

    mlflow.log_metric(
        "f1_weighted",
        f1_weighted,
    )

    mlflow.log_metric(
        "f1_macro",
        f1_macro,
    )

    # MLflow artifact
    mlflow.log_artifact(
        str(confusion_matrix_path),
        artifact_path="evaluation",
    )

    return {
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
    }