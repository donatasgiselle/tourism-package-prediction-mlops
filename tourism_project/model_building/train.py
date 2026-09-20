"""Tune an XGBoost pipeline, track it with MLflow and register it on Hugging Face."""
import os
import json
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from huggingface_hub import HfApi

DATASET_NAME = "tourism-package-prediction"
MODEL_NAME = "tourism-package-model"
MODEL_FILE = "best_tourism_package_model_v1.joblib"
METRICS_FILE = "tourism_project/model_building/model_metrics.json"
CLASSIFICATION_THRESHOLD = 0.45

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-package-prediction")

api = HfApi()
hf_user = api.whoami()["name"]
dataset_repo = f"{hf_user}/{DATASET_NAME}"
model_repo = f"{hf_user}/{MODEL_NAME}"

# Load the train and test data from the Hugging Face data space
base = f"hf://datasets/{dataset_repo}"
Xtrain = pd.read_csv(f"{base}/Xtrain.csv")
Xtest = pd.read_csv(f"{base}/Xtest.csv")
ytrain = pd.read_csv(f"{base}/ytrain.csv").squeeze()
ytest = pd.read_csv(f"{base}/ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_features),
    (make_pipeline(SimpleImputer(strategy="most_frequent"),
                   OneHotEncoder(handle_unknown="ignore")), categorical_features),
)
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss")

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__colsample_bylevel": [0.6, 0.8],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.5, 1.0],
}
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run(run_name="production-training"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all tuned parameter combinations as nested runs
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
    best_model = grid_search.best_estimator_

    # Evaluate with the business threshold
    train_proba = best_model.predict_proba(Xtrain)[:, 1]
    test_proba = best_model.predict_proba(Xtest)[:, 1]
    train_report = classification_report(
        ytrain, (train_proba >= CLASSIFICATION_THRESHOLD).astype(int), output_dict=True)
    test_report = classification_report(
        ytest, (test_proba >= CLASSIFICATION_THRESHOLD).astype(int), output_dict=True)

    metrics = {
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
        "test_roc_auc": roc_auc_score(ytest, test_proba),
    }
    mlflow.log_metrics(metrics)
    print(json.dumps(metrics, indent=2))

    # Save the model locally and keep it as an MLflow artifact
    joblib.dump(best_model, MODEL_FILE)
    mlflow.log_artifact(MODEL_FILE, artifact_path="model")

    # Write a small metrics summary that the workflow commits back to the repository
    with open(METRICS_FILE, "w") as f:
        json.dump({"best_params": grid_search.best_params_,
                   "threshold": CLASSIFICATION_THRESHOLD,
                   "metrics": {k: round(v, 4) for k, v in metrics.items()}}, f, indent=2)

# Register the best model in the Hugging Face model hub
api.create_repo(repo_id=model_repo, repo_type="model", private=False, exist_ok=True)
api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=model_repo,
    repo_type="model",
    commit_message=f"Update model (test F1 = {metrics['test_f1-score']:.3f})",
)
print(f"Model registered at https://huggingface.co/{model_repo}")
