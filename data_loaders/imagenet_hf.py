"""
ImageNet-1K dataset using HuggingFace datasets library.
"""
from datasets import load_dataset
from PIL import Image
from .base import BaseDataset
from .transforms import get_imagenet_transforms


class ImageNet1KDataset(BaseDataset):
    """
    Full ImageNet-1K (1000 classes) dataset using HuggingFace datasets library.
    Loads from ILSVRC/imagenet-1k with caching support.

    Attributes:
        MEAN: Normalization mean values (ImageNet statistics)
        STD: Normalization std values (ImageNet statistics)
        NUM_CLASSES: Number of classes (1000 for ImageNet-1K)
        IMAGE_SIZE: Image dimensions (height, width)
    """

    # ImageNet dataset statistics
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASSES = 1000
    IMAGE_SIZE = (224, 224)

    def __init__(self, train=True, data_dir='/mnt/imagenet/dataset', augmentation='strong', num_classes=None, hf_token=None):
        """
        Initialize ImageNet-1K dataset using HuggingFace datasets.

        Args:
            train: Whether to load train or validation split
            data_dir: Cache directory for HuggingFace dataset (used as cache_dir)
            augmentation: Augmentation strength ('none', 'weak', 'strong')
                         Only applies to training data
            num_classes: Number of classes (fixed at 1000 for ImageNet-1K)
            hf_token: HuggingFace API token for accessing gated datasets
        """
        self.train = train
        self.data_dir = data_dir
        self.augmentation = augmentation if train else 'none'
        self.num_classes = self.NUM_CLASSES  # Always 1000 for ImageNet-1K

        # Warn if num_classes is specified but not 1000
        if num_classes is not None and num_classes != 1000:
            print(f"⚠ Warning: ImageNet-1K always has 1000 classes (specified: {num_classes})")
            print(f"   Using 1000 classes")

        # Load HuggingFace dataset with caching
        print(f"Loading ImageNet-1K dataset from HuggingFace (cache: {data_dir})...")
        try:
            ds = load_dataset("ILSVRC/imagenet-1k", cache_dir=data_dir, token=hf_token)
        except Exception as e:
            if "gated dataset" in str(e) or "authentication" in str(e).lower():
                print(f"\n❌ Error: ImageNet-1K is a gated dataset requiring authentication.")
                print(f"   Please follow these steps:")
                print(f"   1. Request access at: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
                print(f"   2. Get your token at: https://huggingface.co/settings/tokens")
                print(f"   3. Use --hf-token flag: python train.py ... --hf-token YOUR_TOKEN")
                print(f"   OR authenticate once: huggingface-cli login\n")
            raise

        # Select appropriate split
        split_name = 'train' if train else 'validation'
        self.dataset = ds[split_name]

        # Store class names for visualization
        self.class_names = self.dataset.features['label'].names

        print(f"✓ Loaded {len(self.dataset)} samples from {split_name} split")

        # Get appropriate transforms
        self.transform = self.get_transforms(self.augmentation)

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
        # HuggingFace dataset returns dict with 'image' (PIL.Image) and 'label' (int)
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']

        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label

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

        Returns:
            dict: Metadata including num_classes, image_size, mean, std, etc.
        """
        return {
            'name': 'ImageNet-1K (HuggingFace)',
            'num_classes': cls.NUM_CLASSES,
            'image_size': cls.IMAGE_SIZE,
            'mean': cls.MEAN,
            'std': cls.STD,
            'train_samples': 1281167,
            'test_samples': 50000,
            'description': '1000-class ImageNet-1K dataset (ILSVRC2012) via HuggingFace'
        }
