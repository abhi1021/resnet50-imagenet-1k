# ImageNet-1K Training Framework

A modular PyTorch training framework for ImageNet-1K with advanced features including ResNet models (ResNet18/34/50), MixUp augmentation, mixed precision training, and HuggingFace Hub integration.

## Overview

This repository provides a complete, production-ready training pipeline for ImageNet-1K image classification with state-of-the-art techniques:

### Supported Models
- **ResNet-18** (11.7M parameters) - Standard ImageNet architecture
- **ResNet-34** (21.8M parameters) - Deeper standard ResNet
- **ResNet-50** (25.6M parameters) - Bottleneck architecture for ImageNet

### Dataset
- **ImageNet-1K** - 1000 classes, 1.2M training images, 224×224 resolution

### Advanced Features
- **Standard ImageNet data augmentation** (RandomResizedCrop, ColorJitter, etc.)
- **MixUp augmentation** for better generalization
- **Mixed precision training** (FP16) for faster training
- **Learning rate warmup** + Cosine annealing scheduler
- **Label smoothing** and gradient clipping
- **HuggingFace Hub integration** for model versioning
- **Cross-platform GPU support** (CUDA, MPS, CPU)

## Features

✅ **Multi-GPU Support**
- Automatic detection of NVIDIA GPUs (CUDA)
- Apple Silicon GPU support (MPS for M1/M2/M3)
- Graceful CPU fallback

✅ **Advanced Training Techniques**
- Mixed precision training with automatic gradient scaling
- MixUp data augmentation
- Label smoothing
- Gradient clipping
- Warmup + Cosine annealing learning rate schedule

✅ **Modular Architecture**
- Separate modules for data, model, and training
- Easy to extend and customize
- Support for multiple ResNet architectures

✅ **Experiment Tracking**
- Automatic checkpoint management
- Training metrics visualization
- HuggingFace Hub integration for model sharing

✅ **Production Ready**
- Comprehensive CLI interface
- Configurable hyperparameters
- Progress tracking with tqdm
- Console and matplotlib visualizations

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for NVIDIA GPUs) or macOS 12.3+ (for Apple Silicon)

### Setup Development Environment

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/resnet50-imagenet-1k.git
cd resnet50-imagenet-1k
```

2. **Create a virtual environment** (recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n imagenet python=3.10
conda activate imagenet
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## ImageNet Dataset Setup

The ImageNet dataset must be manually downloaded and organized. Follow these steps:

### Download ImageNet

1. **Register and download** from the official ImageNet website:
   - Visit: https://image-net.org/download.php
   - You need to create an account and agree to terms
   - Download ILSVRC2012 (ImageNet Large Scale Visual Recognition Challenge 2012)
   - Files needed: `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`

2. **Alternative sources**:
   - Academic Torrents: https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2
   - Kaggle: https://www.kaggle.com/c/imagenet-object-localization-challenge/data

### Organize Dataset Structure

After downloading, organize the dataset as follows:

```bash
# Create directory structure
mkdir -p /path/to/imagenet/train
mkdir -p /path/to/imagenet/val

# Extract training data
cd /path/to/imagenet/train
tar -xf ILSVRC2012_img_train.tar

# Each training tar file contains one class
for f in *.tar; do
  d=$(basename "$f" .tar)
  mkdir -p "$d"
  tar -xf "$f" -C "$d"
  rm "$f"
done

# Extract validation data
cd /path/to/imagenet/val
tar -xf ILSVRC2012_img_val.tar

# Organize validation images into class folders
# Download the validation ground truth file
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
```

### Expected Directory Structure

```
/path/to/imagenet/
├── train/
│   ├── n01440764/     # Each folder = one class
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ... (1000 classes total)
├── val/
│   ├── n01440764/
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ... (1000 classes total)
```

### Dataset Statistics

- **Total size**: ~150GB (training + validation)
- **Training images**: 1,281,167 images across 1,000 classes
- **Validation images**: 50,000 images (50 per class)
- **Image format**: JPEG
- **Image size**: Variable (will be resized to 224×224 during training)

## Quick Start

### Basic ImageNet Training

Train ResNet-50 on ImageNet-1K with default settings:

```bash
python train.py --data-dir /path/to/imagenet
```

This will:
- Train ResNet-50 on ImageNet for 100 epochs
- Use batch size 256 with all advanced features
- Save checkpoints in `./checkpoints/`
- Display training progress and plots

### Quick Test Run (10 epochs)

```bash
python train.py --data-dir /path/to/imagenet --epochs 10 --batch-size 128
```

## Usage

### Command Line Interface

The training script supports extensive configuration through CLI arguments:

#### Basic Training

```bash
# Train ResNet-50 on ImageNet (default)
python train.py --data-dir /path/to/imagenet

# Train ResNet-18 on ImageNet
python train.py \
  --model resnet18 \
  --data-dir /path/to/imagenet

# Train ResNet-34 on ImageNet
python train.py \
  --model resnet34 \
  --data-dir /path/to/imagenet
```

#### Training Configuration

```bash
# Basic training parameters
python train.py \
  --epochs 90 \
  --batch-size 256 \
  --num-workers 4 \
  --data-dir /path/to/imagenet
```

#### Learning Rate & Optimization

```bash
# Learning rate schedule and optimization settings
python train.py \
  --data-dir /path/to/imagenet \
  --initial-lr 0.01 \
  --max-lr 0.1 \
  --min-lr 1e-4 \
  --warmup-epochs 5 \
  --scheduler cosine \
  --weight-decay 1e-3 \
  --gradient-clip 1.0
```

#### Regularization & Augmentation

```bash
# Configure regularization and data augmentation
python train.py \
  --data-dir /path/to/imagenet \
  --label-smoothing 0.1 \
  --mixup-alpha 0.2

# Or disable MixUp
python train.py --data-dir /path/to/imagenet --no-mixup
```

#### Training Optimizations

```bash
# Disable mixed precision training (enabled by default)
python train.py --data-dir /path/to/imagenet --no-amp
```

#### Checkpointing

```bash
# Configure checkpoint saving
python train.py \
  --data-dir /path/to/imagenet \
  --checkpoint-dir ./checkpoints \
  --checkpoint-epochs 30 60 90
```

#### HuggingFace Integration

```bash
# Enable automatic model upload to HuggingFace Hub
python train.py \
  --data-dir /path/to/imagenet \
  --hf-token YOUR_HF_TOKEN \
  --hf-repo username/imagenet-resnet50
```

## Training Examples

### Example 1: ResNet-50 on ImageNet (Standard)

Standard ResNet-50 training on ImageNet:

```bash
python train.py \
  --model resnet50 \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --checkpoint-epochs 30 60 90
```

### Example 2: ResNet-18 on ImageNet (Faster Training)

Lighter model for faster experimentation:

```bash
python train.py \
  --model resnet18 \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 512
```

### Example 3: Quick Test Run

For testing and debugging:

```bash
python train.py \
  --data-dir /path/to/imagenet \
  --epochs 5 \
  --batch-size 128 \
  --no-amp \
  --checkpoint-epochs 3 5
```

### Example 4: Full Training with HuggingFace Upload

Production training with model versioning:

```bash
python train.py \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --hf-token YOUR_TOKEN \
  --hf-repo username/imagenet-resnet50 \
  --checkpoint-epochs 30 60 90
```

### Example 5: Training Without Advanced Features

Basic training without MixUp or mixed precision:

```bash
python train.py \
  --data-dir /path/to/imagenet \
  --epochs 50 \
  --no-mixup \
  --no-amp \
  --label-smoothing 0.0
```

### Example 6: Apple Silicon (MPS) Optimized

Recommended settings for M1/M2/M3 Macs:

```bash
python train.py \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 128 \
  --num-workers 2 \
  --checkpoint-epochs 30 60 90
```

## CLI Options Reference

### Dataset Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | str | `imagenet` | Dataset to train on (currently only `imagenet`) |
| `--data-dir` | str | `./data` | ImageNet root directory |

### Model Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | str | `resnet50` | Model architecture (`resnet18`, `resnet34`, `resnet50`) |

### Training Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--epochs` | int | `100` | Number of training epochs |
| `--batch-size` | int | `256` | Batch size |
| `--num-workers` | int | `4` | Data loading worker processes |

### Learning Rate Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--initial-lr` | float | `0.01` | Initial learning rate (warmup start) |
| `--max-lr` | float | `0.1` | Maximum learning rate (after warmup) |
| `--min-lr` | float | `1e-4` | Minimum learning rate (cosine annealing) |
| `--warmup-epochs` | int | `5` | Number of warmup epochs |
| `--scheduler` | str | `cosine` | LR scheduler (`cosine`, `onecycle`) |

### Regularization Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--weight-decay` | float | `1e-3` | Weight decay (L2 regularization) |
| `--label-smoothing` | float | `0.1` | Label smoothing factor |
| `--gradient-clip` | float | `1.0` | Gradient clipping max norm |

### Augmentation Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--no-mixup` | flag | False | Disable MixUp augmentation |
| `--mixup-alpha` | float | `0.2` | MixUp interpolation strength |

### Optimization Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--no-amp` | flag | False | Disable mixed precision training |

### Checkpoint Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint-dir` | str | `./checkpoints` | Checkpoint save directory |
| `--checkpoint-epochs` | int[] | `[10, 25, 50, 75]` | Epochs to save checkpoints |

### HuggingFace Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--hf-token` | str | `None` | HuggingFace API token |
| `--hf-repo` | str | `None` | HuggingFace repository ID |

## Project Structure

```
.
├── data.py              # ImageNet dataset loading and transforms
├── train.py             # Training script with ImageClassificationTrainer
├── models/              # Model architectures (ResNet variants)
├── training/            # Training components (optimizer, scheduler, trainer)
├── data_loaders/        # Data loaders and augmentations
├── requirements.txt     # Python dependencies
├── checkpoints/         # Saved model checkpoints (auto-created)
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── checkpoint_epoch*.pth
│   ├── metrics.json
│   ├── training_curves.png
│   └── README.md        # Auto-generated model card
└── data/                # ImageNet dataset directory
```

## Model Architectures

### ResNet-18 (ImageNet)

- **Parameters**: 11.7M
- **Layers**: 18 (2-2-2-2 block configuration)
- **Input**: 224×224 RGB images
- **Output**: 1000 classes

Architecture:
- 7×7 conv, stride 2 → BatchNorm → ReLU → MaxPool
- Residual Block 1: 64 channels (2 basic blocks)
- Residual Block 2: 128 channels (2 basic blocks, stride 2)
- Residual Block 3: 256 channels (2 basic blocks, stride 2)
- Residual Block 4: 512 channels (2 basic blocks, stride 2)
- Global average pooling → FC(512 → 1000)

### ResNet-34 (ImageNet)

- **Parameters**: 21.8M
- **Layers**: 34 (3-4-6-3 block configuration)
- **Input**: 224×224 RGB images
- **Output**: 1000 classes

Similar to ResNet-18 but with more blocks per stage.

### ResNet-50 (ImageNet)

- **Parameters**: 25.6M
- **Layers**: 50 (3-4-6-3 bottleneck configuration)
- **Input**: 224×224 RGB images
- **Output**: 1000 classes

Architecture:
- Uses bottleneck blocks (1×1 → 3×3 → 1×1 convolutions)
- 4× channel expansion in bottleneck
- Residual Block 1: 64→256 channels (3 bottleneck blocks)
- Residual Block 2: 128→512 channels (4 bottleneck blocks, stride 2)
- Residual Block 3: 256→1024 channels (6 bottleneck blocks, stride 2)
- Residual Block 4: 512→2048 channels (3 bottleneck blocks, stride 2)
- Global average pooling → FC(2048 → 1000)

## Data Augmentation

### ImageNet Augmentations

**Training:**
- RandomResizedCrop(224) - Random crop and resize
- RandomHorizontalFlip(p=0.5)
- ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
- Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Validation:**
- Resize(256)
- CenterCrop(224)
- Normalization (ImageNet mean/std)

## Output Files

After training, the following files are created in the checkpoint directory:

- `best_model.pth` - Model with highest validation accuracy
- `final_model.pth` - Model from final epoch
- `checkpoint_epoch{N}.pth` - Checkpoints at specified epochs
- `metrics.json` - Complete training history (losses, accuracies, learning rates)
- `training_curves.png` - Matplotlib plots of training metrics
- `README.md` - Auto-generated model card (if HuggingFace enabled)

### Checkpoint Format

```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'train_accuracy': float,
    'test_accuracy': float,
    'train_loss': float,
    'test_loss': float,
    'best_test_acc': float,
    'timestamp': str,
    'config': dict
}
```

## Loading a Checkpoint

```python
import torch
from models import get_model

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth')

# Create model
model = get_model('resnet50', num_classes=1000)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation accuracy: {checkpoint['test_accuracy']:.2f}%")
```

## HuggingFace Integration

To enable automatic model upload to HuggingFace Hub:

1. **Create a HuggingFace account** at https://huggingface.co

2. **Create an access token**:
   - Go to Settings → Access Tokens
   - Create a token with write permissions

3. **Create a model repository** (optional, auto-created if not exists):
   - Go to your profile → New Model
   - Name it (e.g., `imagenet-resnet50`)

4. **Train with HuggingFace upload**:
```bash
python train.py \
  --data-dir /path/to/imagenet \
  --hf-token YOUR_TOKEN_HERE \
  --hf-repo your-username/imagenet-resnet50
```

Your model will be automatically uploaded to HuggingFace with:
- Checkpoints at specified epochs
- Training metrics
- Training curves
- Auto-generated model card

## GPU Support

The framework automatically detects and uses the best available hardware:

### NVIDIA GPUs (CUDA)
- Full mixed precision support
- Optimized data loading with pin_memory
- Recommended batch size: 256+

### Apple Silicon (MPS)
- M1/M2/M3 GPU acceleration
- Limited mixed precision support (uses fallback)
- Recommended batch size: 128-256
- Recommended workers: 2-4

### CPU
- Works on any system
- Slower training
- Recommended batch size: 64-128

## Performance Tips

### For NVIDIA GPUs
```bash
python train.py --data-dir /path/to/imagenet --batch-size 256 --num-workers 4
```

### For Apple Silicon
```bash
python train.py --data-dir /path/to/imagenet --batch-size 128 --num-workers 2
```

### For CPU Training
```bash
python train.py --data-dir /path/to/imagenet --batch-size 64 --num-workers 2 --no-amp
```

### Reduce Memory Usage
```bash
python train.py --data-dir /path/to/imagenet --batch-size 128 --no-amp
```

## Troubleshooting

### Out of Memory Error
- Reduce `--batch-size` (try 128, 64, or 32)
- Use `--no-amp` to disable mixed precision
- Reduce `--num-workers`

### Slow Training on MPS
- Apple Silicon GPUs work best with batch sizes 128-256
- Try `--num-workers 2` instead of 4
- Mixed precision on MPS has limited support

### Data Loading Issues
- Reduce `--num-workers` (try 2 or 0)
- Check available system memory
- Ensure data directory has read permissions

## Expected Results

### ImageNet (90 epochs)

#### ResNet-18
- **Training Time** (NVIDIA A100): ~24 hours
- **Training Time** (NVIDIA RTX 3090): ~36 hours
- **Expected Top-1 Accuracy**: 69-70%
- **Expected Top-5 Accuracy**: 89-90%

#### ResNet-34
- **Training Time** (NVIDIA A100): ~36 hours
- **Training Time** (NVIDIA RTX 3090): ~48 hours
- **Expected Top-1 Accuracy**: 73-74%
- **Expected Top-5 Accuracy**: 91-92%

#### ResNet-50
- **Training Time** (NVIDIA A100): ~48 hours
- **Training Time** (NVIDIA RTX 3090): ~60-72 hours
- **Expected Top-1 Accuracy**: 75-76%
- **Expected Top-5 Accuracy**: 92-93%

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{imagenet-training,
  author = {Your Name},
  title = {ImageNet-1K Training Framework with ResNet},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/resnet50-imagenet-1k}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- ResNet architecture: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- MixUp augmentation: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- ImageNet dataset: [ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/abs/1409.0575)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
