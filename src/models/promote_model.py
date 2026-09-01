import mlflow
from mlflow import MlflowClient


MODEL_NAME = "CancerRiskModel"

# Minimum acceptable thresholds
MIN_MACRO_F1 = 0.70
MIN_HIGH_RISK_RECALL = 0.90


def main():

    print("=" * 60)
    print("MODEL VALIDATION & PROMOTION")
    print("=" * 60)

    client = MlflowClient()

    # Get latest model version
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise RuntimeError(
            f"No registered versions found for {MODEL_NAME}"
        )

    latest_version = max(
        versions,
        key=lambda v: int(v.version),
    )

    version = latest_version.version

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Version: {version}"
    )

    # Find evaluation run
    experiment = client.get_experiment_by_name(
        "cancer-risk-final-model"
    )

    if experiment is None:
        raise RuntimeError(
            "MLflow experiment not found."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )

    evaluation_run = None

    for run in runs:

        if (
            run.data.tags.get("evaluation_type")
            == "final_test_evaluation"
        ):
            evaluation_run = run
            break

    if evaluation_run is None:
        raise RuntimeError(
            "No final evaluation run found."
        )

    metrics = evaluation_run.data.metrics

    macro_f1 = metrics["f1_macro"]
    high_risk_recall = metrics[
        "high_risk_recall"
    ]

    print(
        f"\nMacro F1: {macro_f1:.4f}"
    )

    print(
        f"High-risk Recall: {high_risk_recall:.4f}"
    )

    print("\nRequired thresholds:")

    print(
        f"Macro F1 >= {MIN_MACRO_F1}"
    )

    print(
        f"High-risk Recall >= "
        f"{MIN_HIGH_RISK_RECALL}"
    )

    # Validation
    passed = (
        macro_f1 >= MIN_MACRO_F1
        and high_risk_recall >= MIN_HIGH_RISK_RECALL
    )

    if passed:

        print("\nVALIDATION PASSED")

        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "validation_status",
            "passed",
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "model_stage",
            "production",
        )

        client.set_registered_model_alias(
            MODEL_NAME,
            "production",
            version,
        )

        print(
            f"\nModel {MODEL_NAME} "
            f"version {version} "
            "promoted to @production."
    )

    else:

        print("\nVALIDATION FAILED")

        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "validation_status",
            "failed",
        )

        raise RuntimeError(
            "Model did not meet production thresholds."
        )


if __name__ == "__main__":
    main()