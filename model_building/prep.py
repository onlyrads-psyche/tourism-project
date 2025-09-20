# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
# prep.py
"""
Data preprocessing script for Tourism Project
- Loads dataset from Hugging Face
- Cleans & preprocesses data
- Performs feature engineering
- Saves train/test splits as CSV
- Uploads processed files to Hugging Face Hub
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from huggingface_hub import HfApi

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_PATH = "hf://datasets/dr-psych/tourism_project/tourism.csv"
REPO_ID = "dr-psych/tourism_project"
REPO_TYPE = "dataset"

# Initialize HF API
api = HfApi(token=HF_TOKEN)

# -------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATASET_PATH)
    print("✅ Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    return df

def preprocess_data(df):
    # Handle missing values
    categorical_cols = [
        'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
        'ProductPitched', 'Designation'
    ]
    numerical_cols = [
        'Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
        'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
        'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups',
        'DurationOfPitch'
    ]

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Binary encoding
    for col in ['Passport', 'OwnCar']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Drop unique identifiers
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)

    # Feature engineering
    df['Income_per_person'] = df['MonthlyIncome'] / (df['NumberOfPersonVisiting'] + 1)
    df['Trips_per_year_ratio'] = df['NumberOfTrips'] / 12
    df['Children_ratio'] = df['NumberOfChildrenVisiting'] / (df['NumberOfPersonVisiting'] + 1)
    df['Followup_per_pitch'] = df['NumberOfFollowups'] / (df['DurationOfPitch'] + 1)

    # Handle infinite values
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Separate features/target
    X = df.drop(['ProdTaken'], axis=1)
    y = df['ProdTaken']

    # Encode categorical
    label_encoders = {}
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numerical features
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=[np.number]).columns
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test

def save_and_upload(X_train, X_test, y_train, y_test):
    files = {
        "X_train.csv": X_train,
        "X_test.csv": X_test,
        "y_train.csv": y_train,
        "y_test.csv": y_test,
    }

    for fname, data in files.items():
        data.to_csv(fname, index=False)
        print(f"📂 Saved {fname}")
        api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"⬆️ Uploaded {fname} to {REPO_ID}")

# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    save_and_upload(X_train, X_test, y_train, y_test)
    print("✅ Preprocessing complete and files uploaded.")
