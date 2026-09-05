import mlflow
import pandas as pd


MODEL_NAME = "CancerRiskModel"
MODEL_URI = f"models:/{MODEL_NAME}@production"

REQUIRED_FEATURES = [
    "Cancer_Type",
    "Age",
    "Gender",
    "Smoking",
    "Alcohol_Use",
    "Obesity",
    "Family_History",
    "Diet_Red_Meat",
    "Diet_Salted_Processed",
    "Fruit_Veg_Intake",
    "Physical_Activity",
    "Air_Pollution",
    "Occupational_Hazards",
    "BRCA_Mutation",
    "H_Pylori_Infection",
    "Calcium_Intake",
    "BMI",
    "Physical_Activity_Level",
]


def load_model():
    """Load the model currently assigned to the production alias."""
    return mlflow.pyfunc.load_model(MODEL_URI)


def validate_input(data: dict):
    """Validate that all required model features are present."""
    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )


def predict(data: dict):
    """
    Predict cancer risk level using the production model.

    Parameters
    ----------
    data : dict
        Patient features required by the production model.

    Returns
    -------
    dict
        Prediction containing the predicted risk level.
    """
    validate_input(data)

    df = pd.DataFrame([data])

    model = load_model()

    prediction = model.predict(df)

    return {
        "risk_level": str(prediction[0])
    }