"""Clean the registered dataset, split it and upload the splits to Hugging Face."""
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

DATASET_NAME = "tourism-package-prediction"
TARGET = "ProdTaken"
SPLIT_FILES = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

api = HfApi()
repo_id = f"{api.whoami()['name']}/{DATASET_NAME}"

# Load the raw data from the Hugging Face dataset repository.
raw_path = hf_hub_download(repo_id=repo_id, filename="tourism.csv", repo_type="dataset")
df = pd.read_csv(raw_path)
print("Loaded raw data:", df.shape)

# Remove columns that carry no predictive information
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Harmonise categorical values
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Remove exact duplicates and records without a target
df = df.drop_duplicates().dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)
print("Cleaned data:", df.shape)

# Keep categorical columns as raw strings; encoding belongs to the model pipeline.
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Stratification preserves the purchase ratio in both splits.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits locally.
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train:", Xtrain.shape, "| Test:", Xtest.shape)
print("Purchase rate train/test:", round(ytrain.mean(), 3), "/", round(ytest.mean(), 3))

# Upload the prepared splits back to the Hugging Face data space.
for file_name in SPLIT_FILES:
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update {file_name}",
    )

print(f"Splits uploaded to https://huggingface.co/datasets/{repo_id}")
