"""
ImageNet dataset with Albumentations transforms.
"""
from torchvision import datasets
from .base import BaseDataset
from .transforms import get_imagenet_transforms


class ImageNetDataset(BaseDataset):
    """
    ImageNet dataset wrapper with Albumentations augmentation.
    Supports both full ImageNet-1K (1000 classes) and subsets like ImageNette (10 classes).

    Attributes:
        MEAN: Normalization mean values (ImageNet statistics)
        STD: Normalization std values (ImageNet statistics)
        NUM_CLASSES: Default number of classes (1000 for full ImageNet-1K)
        IMAGE_SIZE: Image dimensions (height, width)
    """

    # ImageNet dataset statistics
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASSES = 1000
    IMAGE_SIZE = (224, 224)

    def __init__(self, train=True, data_dir='../data/imagenet', augmentation='strong', num_classes=None):
        """
        Initialize ImageNet dataset.

        Args:
            train: Whether to load train or test split
            data_dir: Directory containing ImageNet dataset with train/val folders
            augmentation: Augmentation strength ('none', 'weak', 'strong')
                         Only applies to training data
            num_classes: Number of classes to use (default: 1000 for full ImageNet-1K)
                        Set to 10 for ImageNette/ImageWoof subsets
        """
        self.train = train
        self.data_dir = data_dir
        self.augmentation = augmentation if train else 'none'
        self.num_classes = num_classes if num_classes is not None else self.NUM_CLASSES

        # Get appropriate transforms
        self.transform = self.get_transforms(self.augmentation)

        # Determine split folder name
        split = 'train' if train else 'val'
        dataset_path = f"{self.data_dir}/{split}"

        # Load ImageNet dataset using ImageFolder
        self.dataset = datasets.ImageFolder(
            root=dataset_path,
            transform=self.transform
        )

        # Validate number of classes
        actual_num_classes = len(self.dataset.classes)
        if self.num_classes != actual_num_classes:
            print(f"⚠ Warning: Specified num_classes={self.num_classes} but dataset has {actual_num_classes} classes")
            print(f"   Using actual dataset classes: {actual_num_classes}")
            self.num_classes = actual_num_classes

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
    def get_info(cls, num_classes=None):
        """
        Get dataset metadata.

        Args:
            num_classes: Number of classes (default: 1000 for full ImageNet-1K)

        Returns:
            dict: Metadata including num_classes, image_size, mean, std, etc.
        """
        num_classes = num_classes if num_classes is not None else cls.NUM_CLASSES

        # Adjust sample counts based on dataset type
        if num_classes == 10:
            # ImageNette/ImageWoof approximate counts
            train_samples = 9469  # Approximate for ImageNette
            test_samples = 3925
            description = '10-class ImageNet subset (e.g., ImageNette or ImageWoof)'
        elif num_classes == 1000:
            # Full ImageNet-1K
            train_samples = 1281167
            test_samples = 50000
            description = '1000-class ImageNet-1K dataset (ILSVRC2012)'
        else:
            # Custom subset
            train_samples = -1  # Unknown
            test_samples = -1
            description = f'{num_classes}-class ImageNet subset'

        return {
            'name': 'ImageNet',
            'num_classes': num_classes,
            'image_size': cls.IMAGE_SIZE,
            'mean': cls.MEAN,
            'std': cls.STD,
            'train_samples': train_samples,
            'test_samples': test_samples,
            'description': description
        }
