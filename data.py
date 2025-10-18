from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ImageNet dataset statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------
# ImageNet-1k Data Loading
# ------------------------------
def get_imagenet_train_transforms(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Get ImageNet training transforms with standard augmentation.

    Following standard ImageNet training protocol:
    - RandomResizedCrop to 224x224
    - RandomHorizontalFlip
    - ColorJitter for brightness/contrast/saturation
    - Normalization
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_imagenet_val_transforms(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Get ImageNet validation/test transforms.

    Following standard ImageNet validation protocol:
    - Resize to 256x256
    - CenterCrop to 224x224
    - Normalization
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_imagenet_loaders(data_dir, batch_size=256, num_workers=4, pin_memory=True,
                         persistent_workers=None, prefetch_factor=2):
    """
    Get ImageNet train and validation data loaders.

    Args:
        data_dir: Root directory of ImageNet dataset
                  Expected structure:
                  data_dir/
                      train/
                          n01440764/
                              n01440764_10026.JPEG
                              ...
                          n01443537/
                              ...
                          ...
                      val/
                          n01440764/
                              ILSVRC2012_val_00000293.JPEG
                              ...
                          ...
        batch_size: Batch size for both train and val loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer.
                   Recommended: True for CUDA, False for MPS/CPU
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0)
        prefetch_factor: Number of batches to prefetch per worker (default: 2)

    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)

    Note:
        ImageNet dataset must be manually downloaded and organized.
        See README.md for download instructions.

        For MPS (Apple Silicon):
        - Use num_workers=0-2 for best performance
        - pin_memory=False
        - persistent_workers=False or None
    """
    # Get transforms
    train_transforms = get_imagenet_train_transforms()
    val_transforms = get_imagenet_val_transforms()

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=val_transforms
    )

    # Prepare DataLoader kwargs
    loader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': pin_memory,
    }

    # Add worker-specific settings only if num_workers > 0
    if num_workers > 0:
        loader_kwargs['num_workers'] = num_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor
        if persistent_workers is not None:
            loader_kwargs['persistent_workers'] = persistent_workers
    else:
        loader_kwargs['num_workers'] = 0

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )

    print(f"ImageNet dataset loaded:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Number of classes: {len(train_dataset.classes)}")

    return train_loader, val_loader, train_dataset, val_dataset
