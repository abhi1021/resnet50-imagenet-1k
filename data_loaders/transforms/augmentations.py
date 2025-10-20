"""
Albumentations-based data augmentation transforms.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


class AlbumentationsTransforms:
    """
    Training phase transformations with Albumentations.
    Supports different augmentation strengths.
    """

    def __init__(self, mean, std, augmentation='strong'):
        """
        Initialize augmentation pipeline.

        Args:
            mean: Tuple of mean values for normalization
            std: Tuple of std values for normalization
            augmentation: Augmentation strength ('none', 'weak', 'strong')
        """
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        self.aug = self._build_pipeline()

    def _build_pipeline(self):
        """Build augmentation pipeline based on strength."""
        transforms = []

        if self.augmentation == 'strong':
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.5
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(8, 8),
                    hole_width_range=(8, 8),
                    fill=128,
                    p=0.5
                ),  # Cutout
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
            ])
        elif self.augmentation == 'weak':
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
                    p=0.3
                ),
            ])
        # 'none' augmentation: only normalize and convert to tensor

        # Always add normalization and tensor conversion
        transforms.extend([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

        return A.Compose(transforms)

    def __call__(self, image):
        """Apply augmentation pipeline to image."""
        image = np.array(image)
        return self.aug(image=image)["image"]


class TestTransformWrapper:
    """Test phase transformations (no augmentation, only normalization)."""

    def __init__(self, mean, std):
        """
        Initialize test transforms.

        Args:
            mean: Tuple of mean values for normalization
            std: Tuple of std values for normalization
        """
        self.aug = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def __call__(self, img):
        """Apply test transforms to image."""
        img = np.array(img)
        return self.aug(image=img)["image"]


def get_cifar_transforms(mean, std, train=True, augmentation='strong'):
    """
    Get CIFAR transforms based on train/test mode.

    Args:
        mean: Tuple of mean values for normalization
        std: Tuple of std values for normalization
        train: Whether to get train or test transforms
        augmentation: Augmentation strength for training ('none', 'weak', 'strong')

    Returns:
        Callable transform object
    """
    if train:
        return AlbumentationsTransforms(mean, std, augmentation)
    else:
        return TestTransformWrapper(mean, std)


class ImageNetTransforms:
    """
    ImageNet transformations using Albumentations.
    Follows standard ImageNet training protocol.
    """

    def __init__(self, mean, std, train=True, augmentation='strong'):
        """
        Initialize ImageNet transforms.

        Args:
            mean: Tuple of mean values for normalization
            std: Tuple of std values for normalization
            train: Whether to use train or val transforms
            augmentation: Augmentation strength ('none', 'weak', 'strong')
        """
        self.mean = mean
        self.std = std
        self.train = train
        self.augmentation = augmentation
        self.aug = self._build_pipeline()

    def _build_pipeline(self):
        """Build augmentation pipeline for ImageNet."""
        transforms = []

        if self.train:
            # Training transforms
            transforms.append(A.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), p=1.0))
            transforms.append(A.HorizontalFlip(p=0.5))

            if self.augmentation == 'strong':
                # Strong augmentation: add color jitter
                transforms.extend([
                    A.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                        p=0.8
                    ),
                ])
            elif self.augmentation == 'weak':
                # Weak augmentation: lighter color jitter
                transforms.extend([
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                        p=0.5
                    ),
                ])
            # 'none' augmentation: only RandomResizedCrop and HorizontalFlip
        else:
            # Validation/test transforms
            transforms.extend([
                A.Resize(height=256, width=256),
                A.CenterCrop(height=224, width=224),
            ])

        # Always add normalization and tensor conversion
        transforms.extend([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

        return A.Compose(transforms)

    def __call__(self, image):
        """Apply augmentation pipeline to image."""
        image = np.array(image)
        return self.aug(image=image)["image"]


def get_imagenet_transforms(mean, std, train=True, augmentation='strong'):
    """
    Get ImageNet transforms based on train/val mode.

    Args:
        mean: Tuple of mean values for normalization
        std: Tuple of std values for normalization
        train: Whether to get train or val transforms
        augmentation: Augmentation strength for training ('none', 'weak', 'strong')

    Returns:
        Callable transform object

    Note:
        Training transforms follow standard ImageNet protocol:
        - RandomResizedCrop to 224x224
        - RandomHorizontalFlip
        - ColorJitter (if augmentation is 'weak' or 'strong')

        Validation/test transforms:
        - Resize to 256x256
        - CenterCrop to 224x224
    """
    return ImageNetTransforms(mean, std, train, augmentation)


def mixup_data(x, y, alpha=0.2, device=None):
    """
    Apply MixUp augmentation to inputs and targets.

    Args:
        x: Input batch
        y: Target batch
        alpha: MixUp interpolation strength
        device: Device to use (torch.device or str). If None, uses x's device.

    Returns:
        tuple: (mixed_x, y_a, y_b, lambda) where lambda is the mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)

    # Use input tensor's device if not specified
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
