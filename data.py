import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CIFAR-100 dataset statistics
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)

# ImageNet dataset statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AlbumentationsTransforms:
    """Wrapper for Albumentations transforms to work with torchvision datasets."""

    def __init__(self, mean, std, train=True):
        """
        Initialize transforms for CIFAR-100.

        Args:
            mean: Tuple of mean values for normalization
            std: Tuple of std values for normalization
            train: If True, apply training augmentations. Otherwise, only normalize.
        """
        if train:
            # Training augmentations
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=8,
                    max_width=8,
                    p=0.5,
                    fill_value=tuple([int(x * 255) for x in mean])
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.3
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            # Test/validation transforms (normalize only)
            self.aug = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

    def __call__(self, img):
        """Apply transforms to PIL image."""
        image = np.array(img)
        return self.aug(image=image)["image"]


def get_train_transforms(mean=CIFAR100_MEAN, std=CIFAR100_STD):
    """Get training transforms with data augmentation."""
    return AlbumentationsTransforms(mean=mean, std=std, train=True)


def get_test_transforms(mean=CIFAR100_MEAN, std=CIFAR100_STD):
    """Get test transforms (normalize only)."""
    return AlbumentationsTransforms(mean=mean, std=std, train=False)


def get_cifar100_loaders(data_dir='./data', batch_size=256, num_workers=4, pin_memory=True):
    """
    Get CIFAR-100 train and test data loaders.

    Args:
        data_dir: Directory to store/load CIFAR-100 dataset
        batch_size: Batch size for both train and test loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer.
                   Recommended: True for CUDA, False for MPS/CPU

    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)

    Note:
        pin_memory should be True for CUDA devices (faster data transfer to GPU)
        but False for MPS (Apple Silicon) or CPU devices.
    """
    # Get transforms
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transforms
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transforms
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader, train_dataset, test_dataset


def get_cifar100_classes():
    """Get CIFAR-100 class names."""
    dataset = datasets.CIFAR100(root='./data', train=False, download=False)
    return dataset.classes


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


def get_imagenet_loaders(data_dir, batch_size=256, num_workers=4, pin_memory=True):
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

    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)

    Note:
        ImageNet dataset must be manually downloaded and organized.
        See README.md for download instructions.
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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"ImageNet dataset loaded:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Number of classes: {len(train_dataset.classes)}")

    return train_loader, val_loader, train_dataset, val_dataset
