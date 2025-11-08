"""
Fetch ImageNet-1K class labels.
"""

import json
import urllib.request

def fetch_imagenet_classes(output_path):
    """
    Fetch ImageNet-1K class names and save to file.

    Args:
        output_path: Path to save class names
    """
    print("Fetching ImageNet-1K class names...")

    # Fetch from GitHub (standard ImageNet class index)
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

    print(f"Downloading from: {url}")
    with urllib.request.urlopen(url) as response:
        classes = json.load(response)

    print(f"✓ Fetched {len(classes)} class names")

    # Save to file (one class per line)
    with open(output_path, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    print(f"✓ Saved class names to: {output_path}")

    # Print first 10 classes as sample
    print("\nFirst 10 classes:")
    for i, class_name in enumerate(classes[:10]):
        print(f"  {i}: {class_name}")

    print("\nLast 10 classes:")
    for i, class_name in enumerate(classes[-10:], start=len(classes)-10):
        print(f"  {i}: {class_name}")

    return classes

if __name__ == "__main__":
    output_file = "gradio_app/imagenet_classes.txt"
    fetch_imagenet_classes(output_file)
    print(f"\n✓ ImageNet classes file created successfully!")
