import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from huggingface_hub import HfApi

def load_and_prepare_data():
    # Define constants for the dataset and output paths
    api = HfApi(token=os.getenv("HF_TOKEN"))
    DATASET_PATH = "hf://datasets/dr-psych/tourism_project/tourism.csv"
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # Handle missing values for categorical columns
    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
                          'ProductPitched', 'Designation']

    for col in categorical_cols:
         if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)

    # Handle missing values for numerical columns
    numerical_cols = ['Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
            'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
            'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups',
            'DurationOfPitch']

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Ensure binary columns are properly encoded
    binary_cols = ['Passport', 'OwnCar']
    for col in binary_cols:
        if col in df.columns:
           if df[col].dtype == 'object':
              df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Data validation
    print("\nData validation:")
    print(f"Age range: {df['Age'].min()} - {df['Age'].max()}")
    print(f"CityTier values: {sorted(df['CityTier'].unique())}")
    print(f"Target distribution: {df['ProdTaken'].value_counts()}")

    # Drop Columns
    df.drop(columns=['CustomerID'], inplace=True)

    # Feature engineering
    df['Income_per_person'] = df['MonthlyIncome'] / (df['NumberOfPersonVisiting'] + 1)
    df['Trips_per_year_ratio'] = df['NumberOfTrips'] / 12
    df['Children_ratio'] = np.where(df['NumberOfPersonVisiting'] > 0,
                                   df['NumberOfChildrenVisiting'] / df['NumberOfPersonVisiting'], 0)
    df['Followup_per_pitch'] = df['NumberOfFollowups'] / (df['DurationOfPitch'] + 1)

    # Handle infinite values
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Separate features and target
    X = df.drop(['ProdTaken'], axis=1)
    y = df['ProdTaken']

    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_columns:
         le = LabelEncoder()
         X[col] = le.fit_transform(X[col].astype(str))
         label_encoders[col] = le

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    numerical_features = X.select_dtypes(include=[np.number]).columns

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    # Save files
    X_train_scaled.to_csv("X_train.csv", index=False)
    X_test_scaled.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    # Upload to Hugging Face
    files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for file_path in files:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id="dr-psych/tourism_project",
            repo_type="dataset",
        )

    print("Data preparation completed and uploaded!")

if __name__ == "__main__":
    load_and_prepare_data()
