"""
Dataset factory and registry for managing different datasets.
"""
from .cifar100 import CIFAR100Dataset
from .imagenet import ImageNetDataset
from .imagenet_hf import ImageNet1KDataset

# Dataset registry
DATASETS = {
    'cifar100': CIFAR100Dataset,
    'imagenet': ImageNetDataset,
    'imagenet-1k': ImageNet1KDataset,
    # Future datasets can be added here
    # 'cifar10': CIFAR10Dataset,
}


def get_dataset(name, train=True, data_dir='../data', augmentation='strong', **kwargs):
    """
    Factory function to get dataset by name.

    Args:
        name: Dataset name ('cifar100', 'cifar10', etc.)
        train: Whether to load train or test split
        data_dir: Directory to store/load dataset
        augmentation: Augmentation strength ('none', 'weak', 'strong')
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dataset instance

    Example:
        >>> train_dataset = get_dataset('cifar100', train=True)
        >>> test_dataset = get_dataset('cifar100', train=False, augmentation='none')
    """
    name = name.lower()
    if name not in DATASETS:
        available = ', '.join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    dataset_class = DATASETS[name]
    return dataset_class(train=train, data_dir=data_dir, augmentation=augmentation, **kwargs)


def get_dataset_info(name, num_classes=None):
    """
    Get metadata about a dataset without loading it.

    Args:
        name: Dataset name
        num_classes: Optional number of classes for datasets that support it (e.g., ImageNet subsets)

    Returns:
        dict: Dataset metadata (num_classes, image_size, mean, std, etc.)
    """
    name = name.lower()
    if name not in DATASETS:
        available = ', '.join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    dataset_class = DATASETS[name]

    # Check if the dataset's get_info method accepts num_classes
    if num_classes is not None and name in ['imagenet', 'imagenet-1k']:
        return dataset_class.get_info(num_classes=num_classes)
    else:
        return dataset_class.get_info()


__all__ = ['get_dataset', 'get_dataset_info', 'CIFAR100Dataset', 'ImageNetDataset', 'ImageNet1KDataset']
