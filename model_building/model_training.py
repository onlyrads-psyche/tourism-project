%%writefile tourism_project/model_building/model_training.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

def main():
    # Set MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Tourism_Production_Experiment")

    api = HfApi(token=os.getenv("HF_TOKEN"))

    # Load data
    try:
        X_train = pd.read_csv("X_train.csv")
        X_test = pd.read_csv("X_test.csv")
        y_train = pd.read_csv("y_train.csv")['ProdTaken']
        y_test = pd.read_csv("y_test.csv")['ProdTaken']
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Define features
    numerical_cols = ['Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
                     'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
                     'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups',
                     'DurationOfPitch', 'Income_per_person', 'Trips_per_year_ratio',
                     'Children_ratio', 'Followup_per_pitch']

    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
                       'ProductPitched', 'Designation']

    # Calculate class weight
    class_weight = y_train.value_counts()[0] / y_train.value_counts()[1] if len(y_train.value_counts()) > 1 else 1

    # Create preprocessor
    preprocessor = make_column_transformer(
        (StandardScaler(), numerical_cols),
        (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        remainder='passthrough'
    )

    # Create model pipeline
    model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)
    pipeline = make_pipeline(preprocessor, model)

    # Hyperparameter tuning
    param_grid = {
        'xgbclassifier__n_estimators': [50, 75, 100],
        'xgbclassifier__max_depth': [3, 4, 5],
        'xgbclassifier__learning_rate': [0.01, 0.1, 0.2]
    }

    # MLflow tracking
    with mlflow.start_run():
        search = RandomizedSearchCV(pipeline, param_grid, cv=2, n_jobs=-1, n_iter=5)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Evaluate
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log metrics
        mlflow.log_metrics({
            "accuracy": report['accuracy'],
            "f1_score": report['weighted avg']['f1-score']
        })

        # Save model
        model_path = "best_tourism_model_v1.joblib"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        # Upload to Hugging Face
        repo_id = "dr-psych/tourism_project_model"
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
        except RepositoryNotFoundError:
            create_repo(repo_id=repo_id, repo_type="model", private=False)

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path,
            repo_id=repo_id,
            repo_type="model"
        )

        print("Model training completed and uploaded!")

if __name__ == "__main__":
    main()
