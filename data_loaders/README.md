# Data Loaders

Utilities and dataset wrappers for CIFAR and ImageNet with Albumentations-based transforms.

## Files
- `__init__.py` — dataset registry and factory (`get_dataset`, `get_dataset_info`).
- `base.py` — abstract `BaseDataset` all datasets inherit from.
- `cifar100.py` — CIFAR-100 dataset wrapper with Albumentations transforms.
- `imagenet.py` — local ImageNet (ImageFolder) wrapper with Albumentations transforms.
- `imagenet_hf.py` — ImageNet-1K via Hugging Face Datasets (`ILSVRC/imagenet-1k`).
- `transforms/` — Albumentations pipelines (`augmentations.py`).

## Install
```bash
python -m pip install --upgrade pip
pip install torch torchvision albumentations datasets pillow
```

## Supported datasets (names)
- `cifar100`
- `imagenet` (expects `train/` and `val/` folders under `--data-dir`)
- `imagenet-1k` (via Hugging Face; requires access and auth)

## Authentication (for imagenet-1k)
```bash
# One-time login (preferred)
huggingface-cli login

# Or set an env var and pass it to the loader
export HF_TOKEN={{HF_TOKEN}}
```

## Quick start (Python)
Create PyTorch datasets and loaders using the factory.

```python
from torch.utils.data import DataLoader
from data_loaders import get_dataset, get_dataset_info

# CIFAR-100
train_ds = get_dataset('cifar100', train=True, data_dir='../data', augmentation='strong')
val_ds   = get_dataset('cifar100', train=False, data_dir='../data', augmentation='none')

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)

info = get_dataset_info('cifar100')
print(info)
```

### ImageNet (local ImageFolder)
```python
from data_loaders import get_dataset

train_ds = get_dataset(
    'imagenet',
    train=True,
    data_dir='/path/to/imagenet',  # must contain train/ and val/
    augmentation='strong',         # 'none' | 'weak' | 'strong'
)
val_ds = get_dataset('imagenet', train=False, data_dir='/path/to/imagenet', augmentation='none')
```

### ImageNet-1K (Hugging Face)
Uses the HF Datasets cache directory as `data_dir`.

```python
import os
from data_loaders import get_dataset

train_ds = get_dataset(
    'imagenet-1k',
    train=True,
    data_dir='/mnt/imagenet/dataset',  # HF cache dir (or any path you prefer)
    augmentation='strong',
    hf_token=os.getenv('HF_TOKEN'),     # optional if you ran huggingface-cli login
)
val_ds = get_dataset('imagenet-1k', train=False, data_dir='/mnt/imagenet/dataset')
```

## Using transforms directly (optional)
```python
from data_loaders.transforms import get_cifar_transforms, get_imagenet_transforms

cifar_train_t = get_cifar_transforms(mean=(0.5071,0.4865,0.4409), std=(0.2673,0.2564,0.2761), train=True, augmentation='strong')
imagenet_val_t = get_imagenet_transforms(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), train=False)
```

## Notes
- Augmentation levels: `none`, `weak`, `strong`.
- For ImageNet local, `data_dir` should contain `train/` and `val/` subfolders.
- For ImageNet-1K via HF, ensure you have dataset access and are authenticated.
- Dataloader `num_workers` should be tuned for your machine.
