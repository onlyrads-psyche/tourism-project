from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import sys

def main():
    print("Starting deployment to Hugging Face Space...")
    
    # Check for HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not found")
        sys.exit(1)
    
    # Initialize API
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        print(f"Authenticated as: {user_info.get('name', 'Unknown')}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)
    
    # Using existing space
    repo_id = "dr-psych/tourism_project"  # existing space
    repo_type = "space"
    
    # Check if the space exists (should exist since you created it)
    try:
        space_info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' found. Using existing space.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. This shouldn't happen since you created it.")
        sys.exit(1)
    except Exception as e:
        print(f"Error checking space: {e}")
        sys.exit(1)
    
    # Check if deployment folder exists
    if not os.path.exists("deployment"):
        print("ERROR: deployment folder not found")
        print("Current directory contents:")
        print(os.listdir("."))
        sys.exit(1)
    
    # Upload deployment files to the space
    try:
        print("Uploading deployment files to existing space...")
        api.upload_folder(
            folder_path="deployment",
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"App deployed successfully at: https://huggingface.co/spaces/{repo_id}")
        
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
