"""
Upload CPU checkpoint to HuggingFace Hub
"""

from huggingface_hub import HfApi, login
import os

def upload_checkpoint():
    """Upload CPU checkpoint to HuggingFace Hub."""

    # Repository info
    repo_id = "pandurangpatil/imagenet1k"
    checkpoint_path = "checkpoint_5/best_model_cpu.pth"

    print(f"Uploading checkpoint to HuggingFace Hub...")
    print(f"Repository: {repo_id}")
    print(f"File: {checkpoint_path}")

    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Get file size
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")

    # Initialize HF API
    api = HfApi()

    try:
        # Upload file
        print("\nUploading to HuggingFace Hub...")
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="best_model_cpu.pth",
            repo_id=repo_id,
            repo_type="model",
        )

        print(f"\n✓ Upload successful!")
        print(f"Checkpoint available at: https://huggingface.co/{repo_id}/blob/main/best_model_cpu.pth")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        print("\nPlease make sure you are logged in to HuggingFace:")
        print("  1. Run: huggingface-cli login")
        print("  2. Enter your HuggingFace token")
        print("  3. Re-run this script")
        raise

if __name__ == "__main__":
    upload_checkpoint()
