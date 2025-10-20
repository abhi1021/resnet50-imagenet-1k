"""
ImageNet dataset with flexible transforms support.
"""
from torchvision import datasets
from .base import BaseDataset
from .transforms import get_imagenet_transforms


class ImageNetDataset(BaseDataset):
    """
    ImageNet dataset wrapper with flexible augmentation.

    Supports both ImageNet-1K (1000 classes) and smaller variants like ImageNette.
    Automatically detects number of classes from the dataset directory structure.

    Attributes:
        MEAN: Normalization mean values (ImageNet statistics)
        STD: Normalization std values (ImageNet statistics)
        IMAGE_SIZE: Image dimensions after transforms (height, width)
    """

    # ImageNet dataset statistics
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    IMAGE_SIZE = (224, 224)

    def __init__(self, train=True, data_dir='./data', augmentation='strong', **kwargs):
        """
        Initialize ImageNet dataset.

        Args:
            train: Whether to load train or val split
            data_dir: Root directory of ImageNet dataset
                     Expected structure:
                     data_dir/
                         train/
                             n01440764/
                                 image1.JPEG
                                 ...
                             n01443537/
                                 ...
                         val/
                             n01440764/
                                 image1.JPEG
                                 ...
            augmentation: Augmentation strength ('none', 'weak', 'strong')
                         Only applies to training data
            **kwargs: Additional arguments (unused, for compatibility)

        Note:
            The number of classes is automatically detected from the dataset.
            This allows the same code to work with:
            - Full ImageNet-1K (1000 classes, ~1.28M train images)
            - ImageNette (10 classes, subset of ImageNet)
            - Custom ImageNet subsets
        """
        self.train = train
        self.data_dir = data_dir
        self.augmentation = augmentation if train else 'none'

        # Get appropriate transforms
        self.transform = self.get_transforms(self.augmentation)

        # Determine which split to load
        split_dir = f"{self.data_dir}/train" if self.train else f"{self.data_dir}/val"

        # Load ImageNet dataset using ImageFolder
        # This automatically infers the number of classes from directory structure
        self.dataset = datasets.ImageFolder(
            root=split_dir,
            transform=self.transform
        )

        # Store number of classes (detected from dataset)
        self.num_classes = len(self.dataset.classes)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            tuple: (image_tensor, label)
        """
        return self.dataset[idx]

    def get_transforms(self, augmentation='strong'):
        """
        Get transforms for this dataset.

        Args:
            augmentation: Augmentation strength ('none', 'weak', 'strong')

        Returns:
            Callable transform
        """
        return get_imagenet_transforms(
            mean=self.MEAN,
            std=self.STD,
            train=self.train,
            augmentation=augmentation
        )

    @classmethod
    def get_info(cls):
        """
        Get dataset metadata.

        Returns:
            dict: Metadata including num_classes, image_size, mean, std, etc.

        Note:
            num_classes and sample counts are for full ImageNet-1K.
            For subsets like ImageNette, these will be different at runtime.
        """
        return {
            'name': 'ImageNet',
            'num_classes': 1000,  # Full ImageNet-1K (will be auto-detected at runtime)
            'image_size': cls.IMAGE_SIZE,
            'mean': cls.MEAN,
            'std': cls.STD,
            'train_samples': 1281167,  # Full ImageNet-1K
            'test_samples': 50000,     # Full ImageNet-1K
            'description': 'Large-scale image classification dataset (supports full ImageNet-1K and subsets like ImageNette)'
        }
