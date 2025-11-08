"""
Upload the CPU-optimized checkpoint to HuggingFace Hub.
This uploads only the best_model_cpu.pth file.
"""

from huggingface_hub import HfApi
import os
import sys

def upload_cpu_checkpoint(hf_token, repo_id="pandurangpatil/imagenet1k"):
    """
    Upload CPU checkpoint to HuggingFace Hub.

    Args:
        hf_token: HuggingFace API token
        repo_id: Repository ID (default: pandurangpatil/imagenet1k)
    """
    checkpoint_path = "checkpoint_5/best_model_cpu.pth"

    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Get file size
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB

    print("="*70)
    print("Upload CPU Checkpoint to HuggingFace Hub")
    print("="*70)
    print(f"\nRepository: {repo_id}")
    print(f"File: {checkpoint_path}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Target filename: best_model_cpu.pth")

    try:
        # Initialize API
        api = HfApi(token=hf_token)

        # Upload file
        print("\nUploading...")
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="best_model_cpu.pth",
            repo_id=repo_id,
            repo_type="model",
        )

        print("\n" + "="*70)
        print("✓ Upload successful!")
        print(f"View at: https://huggingface.co/{repo_id}/blob/main/best_model_cpu.pth")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        print("\nMake sure you are logged in:")
        print("  huggingface-cli login")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload CPU checkpoint to HuggingFace Hub")
    parser.add_argument("--hf-token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--hf-repo", type=str, default="pandurangpatil/imagenet1k", help="Repository ID")

    args = parser.parse_args()
    upload_cpu_checkpoint(args.hf_token, args.hf_repo)
