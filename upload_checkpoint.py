"""
Standalone script to upload a trained model checkpoint to HuggingFace Hub.

This script allows you to upload the best model from a checkpoint folder to HuggingFace
without needing to retrain. It will:
1. Validate that best_model.pth exists in the checkpoint folder
2. Optionally regenerate the model card (README.md) from checkpoint metadata
3. Upload all available files to HuggingFace Hub

Usage:
    python upload_checkpoint.py \
        --checkpoint-dir checkpoint_5 \
        --hf-token YOUR_TOKEN \
        --hf-repo username/model-name \
        --regenerate-readme
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Import utilities from the project
from utils.huggingface import HuggingFaceUploader, create_model_card
from utils.metrics import MetricsTracker


def validate_checkpoint_dir(checkpoint_dir):
    """
    Validate that checkpoint directory exists and contains best_model.pth.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Path: Absolute path to checkpoint directory

    Raises:
        SystemExit: If validation fails
    """
    checkpoint_path = Path(checkpoint_dir)

    # Check if directory exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    if not checkpoint_path.is_dir():
        print(f"Error: Path is not a directory: {checkpoint_dir}")
        sys.exit(1)

    # Check for best_model.pth
    best_model_path = checkpoint_path / "best_model.pth"
    if not best_model_path.exists():
        print(f"Error: best_model.pth not found in {checkpoint_dir}")
        print(f"Expected location: {best_model_path}")
        print("\nThe checkpoint directory must contain best_model.pth")
        sys.exit(1)

    print(f"✓ Found best_model.pth in {checkpoint_dir}")
    return checkpoint_path.absolute()


def load_checkpoint_metadata(checkpoint_dir):
    """
    Load config.json and metrics.json from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        tuple: (config_dict, metrics_tracker) or (None, None) if files missing
    """
    config_path = checkpoint_dir / "config.json"
    metrics_path = checkpoint_dir / "metrics.json"

    config = None
    metrics_tracker = None

    # Load config.json
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Loaded config.json")
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
    else:
        print(f"Warning: config.json not found in {checkpoint_dir}")

    # Load metrics.json
    if metrics_path.exists():
        try:
            metrics_tracker = MetricsTracker()
            metrics_tracker.load(str(metrics_path))
        except Exception as e:
            print(f"Warning: Could not load metrics.json: {e}")
            metrics_tracker = None
    else:
        print(f"Warning: metrics.json not found in {checkpoint_dir}")

    return config, metrics_tracker


def generate_readme(checkpoint_dir, repo_id, config, metrics_tracker):
    """
    Generate README.md model card from checkpoint metadata.

    Args:
        checkpoint_dir: Path to checkpoint directory
        repo_id: HuggingFace repository ID
        config: Configuration dictionary
        metrics_tracker: MetricsTracker instance

    Returns:
        bool: True if README.md was generated successfully, False otherwise
    """
    if config is None or metrics_tracker is None:
        print("Warning: Cannot regenerate README.md - missing config.json or metrics.json")
        return False

    try:
        # Extract metadata from config and metrics
        model_name = config.get('model', 'Unknown')
        best_acc = metrics_tracker.best_test_acc
        total_epochs = len(metrics_tracker.test_accuracies)

        # Generate model card
        model_card_content = create_model_card(
            repo_id=repo_id,
            model_name=model_name,
            best_acc=best_acc,
            total_epochs=total_epochs,
            metrics_tracker=metrics_tracker,
            config=config
        )

        # Save README.md
        readme_path = checkpoint_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(model_card_content)

        print(f"✓ Generated README.md")
        return True

    except Exception as e:
        print(f"Warning: Could not generate README.md: {e}")
        return False


def check_available_files(checkpoint_dir):
    """
    Check which files are available in the checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        list: List of filenames that exist
    """
    standard_files = [
        'best_model.pth',
        'config.json',
        'metrics.json',
        'training_curves.png',
        'lr_finder_plot.png',
        'README.md'
    ]

    available_files = []
    for filename in standard_files:
        if (checkpoint_dir / filename).exists():
            available_files.append(filename)

    return available_files


def main():
    """Main entry point for the upload script."""
    parser = argparse.ArgumentParser(
        description='Upload best model checkpoint to HuggingFace Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload checkpoint with regenerated model card
  python upload_checkpoint.py \\
      --checkpoint-dir checkpoint_5 \\
      --hf-token YOUR_TOKEN \\
      --hf-repo username/resnet50-imagenet-1k

  # Upload without regenerating README.md
  python upload_checkpoint.py \\
      --checkpoint-dir checkpoint_5 \\
      --hf-token YOUR_TOKEN \\
      --hf-repo username/resnet50-imagenet-1k \\
      --no-regenerate-readme

For more information, see: https://huggingface.co/docs/huggingface_hub
        """
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='Path to checkpoint directory containing best_model.pth'
    )

    parser.add_argument(
        '--hf-token',
        type=str,
        required=True,
        help='HuggingFace API token (get from https://huggingface.co/settings/tokens)'
    )

    parser.add_argument(
        '--hf-repo',
        type=str,
        required=True,
        help='HuggingFace repository ID (e.g., username/model-name)'
    )

    parser.add_argument(
        '--regenerate-readme',
        action='store_true',
        default=True,
        help='Regenerate README.md from checkpoint metadata (default: True)'
    )

    parser.add_argument(
        '--no-regenerate-readme',
        action='store_false',
        dest='regenerate_readme',
        help='Do not regenerate README.md, use existing file if present'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("HuggingFace Checkpoint Upload Tool")
    print("="*70)

    # Step 1: Validate checkpoint directory
    print("\n[1/4] Validating checkpoint directory...")
    checkpoint_dir = validate_checkpoint_dir(args.checkpoint_dir)

    # Step 2: Load checkpoint metadata
    print("\n[2/4] Loading checkpoint metadata...")
    config, metrics_tracker = load_checkpoint_metadata(checkpoint_dir)

    # Step 3: Regenerate README.md if requested
    print("\n[3/4] Preparing model card...")
    if args.regenerate_readme:
        print("Attempting to regenerate README.md...")
        generate_readme(checkpoint_dir, args.hf_repo, config, metrics_tracker)
    else:
        print("Skipping README.md regeneration (--no-regenerate-readme)")
        readme_path = checkpoint_dir / "README.md"
        if readme_path.exists():
            print(f"✓ Will use existing README.md")
        else:
            print(f"Warning: README.md not found, no model card will be uploaded")

    # Step 4: Check available files and upload
    print("\n[4/4] Uploading to HuggingFace Hub...")
    available_files = check_available_files(checkpoint_dir)

    print(f"\nFiles to upload:")
    for filename in available_files:
        file_size = (checkpoint_dir / filename).stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"  - {filename} ({size_mb:.2f} MB)")

    try:
        # Initialize uploader
        uploader = HuggingFaceUploader(
            repo_id=args.hf_repo,
            token=args.hf_token
        )

        # Upload all files
        uploader.upload_checkpoint_files(str(checkpoint_dir))

        print("\n" + "="*70)
        print("Upload completed successfully!")
        print(f"View your model at: https://huggingface.co/{args.hf_repo}")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nError during upload: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your HuggingFace token is valid")
        print("2. Check that you have write access to the repository")
        print("3. Ensure huggingface_hub is installed: pip install huggingface_hub")
        sys.exit(1)


if __name__ == '__main__':
    main()
