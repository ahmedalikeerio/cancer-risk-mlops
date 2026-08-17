from datasets import load_dataset


DATASET_ID = "tarekmasryo/cancer-risk-factors-data"

EXPECTED_COLUMNS = {
    "Patient_ID",
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
    "Overall_Risk_Score",
    "BMI",
    "Physical_Activity_Level",
    "Risk_Level",
}

RISK_LEVELS = {"Low", "Medium", "High"}

SCORE_COLUMNS = [
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
    "Physical_Activity_Level",
]


def validate_dataset():
    dataset = load_dataset(DATASET_ID)
    df = dataset["train"].to_pandas()

    errors = []

    # 1. Schema validation
    actual_columns = set(df.columns)

    missing_columns = EXPECTED_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_COLUMNS

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    if unexpected_columns:
        errors.append(f"Unexpected columns: {unexpected_columns}")

    # 2. Missing values
    missing_values = df.isnull().sum()

    if missing_values.any():
        errors.append(
            f"Missing values found:\n"
            f"{missing_values[missing_values > 0]}"
        )

    # 3. Duplicate rows
    duplicate_rows = df.duplicated().sum()

    if duplicate_rows > 0:
        errors.append(
            f"Duplicate rows found: {duplicate_rows}"
        )

    # 4. Duplicate patient IDs
    duplicate_ids = df["Patient_ID"].duplicated().sum()

    if duplicate_ids > 0:
        errors.append(
            f"Duplicate Patient_ID values: {duplicate_ids}"
        )

    # 5. Target validation
    actual_risk_levels = set(df["Risk_Level"].unique())

    unexpected_risk_levels = actual_risk_levels - RISK_LEVELS

    if unexpected_risk_levels:
        errors.append(
            f"Unexpected Risk_Level values: "
            f"{unexpected_risk_levels}"
        )

    # 6. Age validation
    if not df["Age"].between(25, 90).all():
        errors.append("Age contains values outside the observed 25–90 range.")

    # 7. BMI validation
    if not df["BMI"].between(15, 41.4).all():
        errors.append("BMI contains values outside the observed 15–41.4 range.")

    # 8. Risk-factor score validation
    for column in SCORE_COLUMNS:
        if not df[column].between(0, 10).all():
            errors.append(
                f"{column} contains values outside the expected 0–10 range."
            )

    # Final result
    if errors:
        print("DATASET VALIDATION FAILED\n")

        for error in errors:
            print(f"- {error}")

        raise ValueError("Dataset validation failed.")

    print(" DATASET VALIDATION PASSED")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Risk levels: {sorted(df['Risk_Level'].unique())}")


if __name__ == "__main__":
    validate_dataset()