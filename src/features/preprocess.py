from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump


RAW_DATA_PATH = Path("data/raw/cancer_risk.csv")
PROCESSED_DATA_PATH = Path("data/processed/cancer_risk_processed.csv")
PREPROCESSOR_PATH = Path("data/processed/preprocessor.joblib")

TARGET = "Risk_Level"

DROP_COLUMNS = [
    'Patient_ID',
    'Overall_Risk_Score'
]

def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)

    X= df.drop(columns=[TARGET] + DROP_COLUMNS)
    y= df[TARGET]

    categorical_columns = X.select_dtypes(include=['object','category']).columns.tolist()
    numerical_columns = X.select_dtypes(include='number').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers =[('numerical','passthrough',numerical_columns),
                       ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical_columns)]
    )

    x_processed = preprocessor.fit_transform(X)
    feature_names= preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(x_processed,columns=feature_names)
    processed_df[TARGET] = y.values

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(PROCESSED_DATA_PATH,index=False)

    dump(preprocessor, PREPROCESSOR_PATH)

    print("PREPROCESSING COMPLETED")
    print(f'original shape {df.shape}')
    print(f'Processed shape: {processed_df.shape}')
    print(f'saved dataset: {PROCESSED_DATA_PATH}')
    print(f'Saved processor: {PREPROCESSOR_PATH}')


if __name__ == '__main__':
    preprocess_data()
