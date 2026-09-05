from src.inference.predict import predict


def test_prediction_returns_valid_risk_level():
    patient_data = {
        "Cancer_Type": "Prostate",
        "Age": 72,
        "Gender": 1,
        "Smoking": 1,
        "Alcohol_Use": 4,
        "Obesity": 9,
        "Family_History": 0,
        "Diet_Red_Meat": 1,
        "Diet_Salted_Processed": 7,
        "Fruit_Veg_Intake": 5,
        "Physical_Activity": 10,
        "Air_Pollution": 7,
        "Occupational_Hazards": 6,
        "BRCA_Mutation": 0,
        "H_Pylori_Infection": 0,
        "Calcium_Intake": 7,
        "BMI": 26.7,
        "Physical_Activity_Level": 0,
    }

    result = predict(patient_data)

    assert isinstance(result, dict)
    assert "risk_level" in result

    assert result["risk_level"] in {
        "Low",
        "Medium",
        "High",
    }


def test_missing_features_are_rejected():
    patient_data = {
        "Cancer_Type": "Prostate",
        "Age": 72,
    }

    try:
        predict(patient_data)
        assert False, "Expected ValueError for missing features"
    except ValueError as exc:
        assert "Missing required features" in str(exc)