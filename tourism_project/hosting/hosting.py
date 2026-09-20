"""Create or update the Hugging Face Space and push all deployment files to it."""
from huggingface_hub import HfApi

SPACE_NAME = "tourism-package-app"
MODEL_NAME = "tourism-package-model"

api = HfApi()
hf_user = api.whoami()["name"]
space_id = f"{hf_user}/{SPACE_NAME}"

# Create a Docker-based Space if it does not exist yet
api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker",
                private=False, exist_ok=True)

# Tell the app which model repository to load
api.add_space_variable(repo_id=space_id, key="MODEL_REPO_ID", value=f"{hf_user}/{MODEL_NAME}")

# Upload Dockerfile, app.py, requirements.txt and README.md
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=space_id,
    repo_type="space",
    ignore_patterns=["__pycache__/*", "*.pyc"],
    commit_message="Deploy Streamlit app",
)
print(f"Space updated: https://huggingface.co/spaces/{space_id}")
