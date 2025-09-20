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

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/dr-psych/tourism_project/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Handle missing values
# Fill missing values for categorical columns with mode
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
                      'ProductPitched', 'Designation']

for col in categorical_cols:
     if col in df.columns and df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Fill missing values for numerical columns with median
numerical_cols = ['Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
        'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
        'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups',
        'DurationOfPitch']

for col in numerical_cols:
    if col in df.columns and df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Ensure binary columns are properly encoded (0/1)
binary_cols = ['Passport', 'OwnCar']
for col in binary_cols:
    if col in df.columns:
       # Convert Yes/No to 1/0 if needed
       if df[col].dtype == 'object':
          df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

# Validate data types and ranges
print("\nData validation:")
print(f"Age range: {df['Age'].min()} - {df['Age'].max()}")
print(f"CityTier values: {sorted(df['CityTier'].unique())}")
print(f"MonthlyIncome range: {df['MonthlyIncome'].min()} - {df['MonthlyIncome'].max()}")
print(f"Target distribution: {df['ProdTaken'].value_counts()}")

return df

# Ensure all required columns are present (based on data description)
required_columns = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
        'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar',
        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',
        'PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups',
        'DurationOfPitch'
    ]

# Check for missing columns
missing_cols = [col for col in required_columns if col not in df.columns]
  if missing_cols:
     print(f"Warning: Missing columns: {missing_cols}")


# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)


# Create feature engineering
df['Income_per_person'] = df['MonthlyIncome'] / (df['NumberOfPersonVisiting'] + 1)
df['Trips_per_year_ratio'] = df['NumberOfTrips'] / 12
df['Children_ratio'] = df['NumberOfChildrenVisiting'] / df['NumberOfPersonVisiting']
df['Followup_per_pitch'] = df['NumberOfFollowups'] / (df['DurationOfPitch'] + 1)


# Handle infinite values
df.replace([np.inf, -np.inf], 0, inplace=True)


# Separate features and target
X = df.drop(['ProdTaken'], axis=1)
y = df['ProdTaken']


# Encode categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns

for col in categorical_columns:
     le = LabelEncoder()
     X[col] = le.fit_transform(X[col].astype(str))
     label_encoders[col] = le


# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_features = X.select_dtypes(include=[np.number]).columns

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders


X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)


files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="dr-psych/tourism_project",
        repo_type="dataset",
    )
