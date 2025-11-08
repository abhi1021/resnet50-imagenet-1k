"""
Convert PyTorch checkpoint to CPU-compatible format.
This script extracts only the model weights and removes optimizer state
to reduce file size and ensure CPU compatibility.
"""

import torch
import os

def convert_checkpoint_to_cpu(input_path, output_path):
    """
    Convert checkpoint to CPU-compatible format.

    Args:
        input_path: Path to original checkpoint
        output_path: Path to save CPU-compatible checkpoint
    """
    print(f"Loading checkpoint from: {input_path}")

    # Load checkpoint with CPU mapping
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    print(f"Original checkpoint keys: {list(checkpoint.keys())}")
    print(f"Checkpoint info:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Test Accuracy: {checkpoint.get('test_accuracy', 'N/A')}%")
    print(f"  - Train Accuracy: {checkpoint.get('train_accuracy', 'N/A')}%")

    # Extract only model state dict for smaller file size
    cpu_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        'epoch': checkpoint.get('epoch', 0),
        'test_accuracy': checkpoint.get('test_accuracy', 0.0),
        'train_accuracy': checkpoint.get('train_accuracy', 0.0),
        'test_loss': checkpoint.get('test_loss', 0.0),
        'config': checkpoint.get('config', {})
    }

    # Save CPU-compatible checkpoint
    print(f"\nSaving CPU checkpoint to: {output_path}")
    torch.save(cpu_checkpoint, output_path)

    # Get file sizes
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    new_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

    print(f"\nFile size comparison:")
    print(f"  - Original: {original_size:.2f} MB")
    print(f"  - CPU-optimized: {new_size:.2f} MB")
    print(f"  - Reduction: {original_size - new_size:.2f} MB ({((original_size - new_size) / original_size * 100):.1f}%)")

    # Verify the checkpoint can be loaded
    print("\nVerifying checkpoint can be loaded...")
    test_load = torch.load(output_path, map_location='cpu', weights_only=False)
    print(f"✓ Checkpoint verified. Keys: {list(test_load.keys())}")

    return cpu_checkpoint

if __name__ == "__main__":
    # Paths
    input_checkpoint = "checkpoint_5/best_model.pth"
    output_checkpoint = "checkpoint_5/best_model_cpu.pth"

    # Convert
    convert_checkpoint_to_cpu(input_checkpoint, output_checkpoint)

    print("\n✓ Conversion complete!")
    print(f"CPU-compatible checkpoint saved to: {output_checkpoint}")
