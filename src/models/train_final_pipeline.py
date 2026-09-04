from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Paths
# ============================================================

TRAIN_PATH = Path(
    "data/split/train.csv"
)

MODEL_PATH = Path(
    "models/final_pipeline.joblib"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "Risk_Level"

DROP_COLUMNS = [
    "Patient_ID",
    "Overall_Risk_Score",
]

MODEL_NAME = "CancerRiskModel"

EXPERIMENT_NAME = "cancer-risk-final-model"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("FINAL PIPELINE MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    X_train = train_df.drop(
        columns=[TARGET] + DROP_COLUMNS
    )

    y_train = train_df[TARGET]

    print(
        f"Training data: {X_train.shape}"
    )

    print(
        f"Classes: {sorted(y_train.unique())}"
    )

    # --------------------------------------------------------
    # Identify feature types
    # --------------------------------------------------------

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print(
        f"Numerical features: {len(numerical_columns)}"
    )

    print(
        f"Categorical features: {len(categorical_columns)}"
    )

    print(
        f"Numerical columns: {numerical_columns}"
    )

    print(
        f"Categorical columns: {categorical_columns}"
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
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

    # --------------------------------------------------------
    # Complete ML pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    C=100.0,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="final_logistic_regression_pipeline"
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Generate MLflow signature
        # ----------------------------------------------------

        input_example = X_train.head(3)

        example_predictions = pipeline.predict(
            input_example
        )

        signature = infer_signature(
            X_train,
            example_predictions,
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "C": 100.0,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42,
        })

        mlflow.log_params({
            "training_rows": len(X_train),
            "features": X_train.shape[1],
            "numerical_features": len(
                numerical_columns
            ),
            "categorical_features": len(
                categorical_columns
            ),
        })

        # ----------------------------------------------------
        # Log and register complete pipeline
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
            signature=signature,
        )

        # ----------------------------------------------------
        # Save local pipeline
        # ----------------------------------------------------

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            pipeline,
            MODEL_PATH,
        )

        print(
            f"\nSaved pipeline: {MODEL_PATH}"
        )

        print(
            f"Registered model: {MODEL_NAME}"
        )

    print(
        "\nFINAL PIPELINE MODEL TRAINED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()