from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Attaching Huggingface repo id and repo typ
repo_id = "dr-psych/tourism_project"
repo_type = "dataset"

# Stating HF token environment and stating the secret key
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Hugging Face token (HF_TOKEN) is not set as an environment variable.")

from huggingface_hub import HfApi
api = HfApi()
# Pass the token explicitly when creating or interacting with repos
# api.create_repo(repo_id=repo_id, token=HF_TOKEN, repo_type="dataset")

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")


api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)
