from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Different repo for the space (app deployment)
repo_id = "dr-psych/tourism_project"  # Different name for space
repo_type = "space"  # Space for Streamlit app, not dataset

# Initialize API client with environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="streamlit",  # Specify it's a Streamlit space
        private=False
    )
    print(f"Space '{repo_id}' created.")

# Upload deployment files to the space
api.upload_folder(
    folder_path="deployment",
    repo_id=repo_id,
    repo_type=repo_type,
)

print(f"App deployed successfully at: https://huggingface.co/spaces/{repo_id}")
