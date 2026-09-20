"""Register the raw tourism dataset in a Hugging Face dataset repository."""
import os
import pandas as pd
from huggingface_hub import HfApi

DATA_DIR = "tourism_project/data"
RAW_PATH = os.path.join(DATA_DIR, "tourism.csv")
DATASET_NAME = "tourism-package-prediction"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

# Validate the schema before anything is published
df = pd.read_csv(RAW_PATH)
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")
print(f"Schema check passed: {df.shape[0]} rows, {df.shape[1]} columns")

# The token is taken from the HF_TOKEN environment variable
api = HfApi()
repo_id = f"{api.whoami()['name']}/{DATASET_NAME}"

# Create the dataset repository if necessary and upload the data folder
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(
    folder_path=DATA_DIR,
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["tourism.csv"],
    commit_message="Register raw tourism dataset",
)
print(f"Dataset registered at https://huggingface.co/datasets/{repo_id}")
