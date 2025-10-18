# CIFAR-100 Training Framework

A modular PyTorch training framework for CIFAR-100 with advanced features including WideResNet-28-10, MixUp augmentation, mixed precision training, and HuggingFace Hub integration.

## Overview

This repository provides a complete, production-ready training pipeline for CIFAR-100 image classification with state-of-the-art techniques:

- **WideResNet-28-10** architecture (36.5M parameters)
- **Advanced data augmentation** with Albumentations
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
- Support for multiple model architectures

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
conda create -n cifar100 python=3.10
conda activate cifar100
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Basic Training

Train WideResNet-28-10 on CIFAR-100 with default settings:

```bash
python train.py
```

This will:
- Download CIFAR-100 dataset automatically
- Train for 100 epochs with batch size 256
- Use all advanced features (MixUp, mixed precision, etc.)
- Save checkpoints in `./checkpoints/`
- Display training progress and plots

### Simple Training (10 epochs)

```bash
python train.py --epochs 10 --batch-size 128
```

## Usage

### Command Line Interface

The training script supports extensive configuration through CLI arguments:

#### Model Configuration

```bash
python train.py \
  --model wideresnet \           # Model architecture: 'wideresnet' or 'net'
  --depth 28 \                   # WideResNet depth (default: 28)
  --widen-factor 10 \            # WideResNet width multiplier (default: 10)
  --dropout 0.3                  # Dropout rate (default: 0.3)
```

#### Training Configuration

```bash
python train.py \
  --epochs 100 \                 # Number of epochs (default: 100)
  --batch-size 256 \             # Batch size (default: 256)
  --num-workers 4 \              # Data loading workers (default: 4)
  --data-dir ./data              # Data directory (default: ./data)
```

#### Learning Rate & Optimization

```bash
python train.py \
  --initial-lr 0.01 \            # Initial LR for warmup (default: 0.01)
  --max-lr 0.1 \                 # Max LR after warmup (default: 0.1)
  --min-lr 1e-4 \                # Min LR for cosine annealing (default: 1e-4)
  --warmup-epochs 5 \            # Warmup epochs (default: 5)
  --scheduler cosine \           # Scheduler: 'cosine' or 'onecycle'
  --weight-decay 1e-3 \          # Weight decay (default: 1e-3)
  --gradient-clip 1.0            # Gradient clipping max norm (default: 1.0)
```

#### Regularization & Augmentation

```bash
python train.py \
  --label-smoothing 0.1 \        # Label smoothing (default: 0.1)
  --mixup-alpha 0.2 \            # MixUp alpha (default: 0.2)
  --no-mixup                     # Disable MixUp augmentation
```

#### Training Optimizations

```bash
python train.py \
  --no-amp                       # Disable mixed precision training
```

#### Checkpointing

```bash
python train.py \
  --checkpoint-dir ./checkpoints \           # Checkpoint directory
  --checkpoint-epochs 10 25 50 75            # Save at specific epochs
```

#### HuggingFace Integration

```bash
python train.py \
  --hf-token YOUR_HF_TOKEN \                 # HuggingFace API token
  --hf-repo username/cifar100-model          # HuggingFace repo ID
```

## Training Examples

### Example 1: Quick Test Run (Fast Training)

For testing and debugging:

```bash
python train.py \
  --epochs 5 \
  --batch-size 128 \
  --no-amp \
  --checkpoint-epochs 3 5
```

### Example 2: Standard Training

Recommended settings for good performance:

```bash
python train.py \
  --epochs 100 \
  --batch-size 256 \
  --mixup-alpha 0.2 \
  --label-smoothing 0.1 \
  --checkpoint-epochs 25 50 75 100
```

### Example 3: Full Training with HuggingFace Upload

Production training with model versioning:

```bash
python train.py \
  --epochs 100 \
  --batch-size 256 \
  --hf-token YOUR_TOKEN \
  --hf-repo username/cifar100-wideresnet \
  --checkpoint-epochs 10 25 50 75
```

### Example 4: Training Without Advanced Features

Basic training without MixUp or mixed precision:

```bash
python train.py \
  --epochs 50 \
  --no-mixup \
  --no-amp \
  --label-smoothing 0.0
```

### Example 5: Legacy CNN Model

Train the smaller custom CNN instead of WideResNet:

```bash
python train.py \
  --model net \
  --epochs 30 \
  --batch-size 128 \
  --scheduler onecycle
```

### Example 6: Apple Silicon (MPS) Optimized

Recommended settings for M1/M2/M3 Macs:

```bash
python train.py \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 2 \
  --checkpoint-epochs 25 50 75
```

## CLI Options Reference

### Model Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | str | `wideresnet` | Model architecture (`wideresnet`, `net`) |
| `--depth` | int | `28` | WideResNet depth |
| `--widen-factor` | int | `10` | WideResNet width multiplier |
| `--dropout` | float | `0.3` | Dropout rate |

### Training Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--epochs` | int | `100` | Number of training epochs |
| `--batch-size` | int | `256` | Batch size |
| `--data-dir` | str | `./data` | CIFAR-100 data directory |
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
├── data.py              # CIFAR-100 dataset loading and transforms
├── model.py             # Model architectures (WideResNet, CNN)
├── train.py             # Training script with CIFAR100Trainer class
├── requirements.txt     # Python dependencies
├── checkpoints/         # Saved model checkpoints (auto-created)
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── checkpoint_epoch*.pth
│   ├── metrics.json
│   ├── training_curves.png
│   └── README.md        # Auto-generated model card
└── data/                # CIFAR-100 dataset (auto-downloaded)
```

## Model Architecture

### WideResNet-28-10

- **Parameters**: 36.5M
- **Depth**: 28 layers
- **Width Factor**: 10
- **Dropout**: 0.3
- **Input**: 32×32 RGB images
- **Output**: 100 classes

Architecture details:
- Initial conv: 3→16 channels
- Block 1: 16→160 channels (4 residual blocks)
- Block 2: 160→320 channels (4 residual blocks, stride 2)
- Block 3: 320→640 channels (4 residual blocks, stride 2)
- Global average pooling
- Fully connected: 640→100

### Legacy CNN (Net)

Smaller custom CNN for CIFAR-10 compatibility (backward compatible).

## Data Augmentation

### Training Augmentations (Albumentations)

- HorizontalFlip (p=0.5)
- ShiftScaleRotate (shift=0.0625, scale=0.1, rotate=15°)
- CoarseDropout/Cutout (8×8 holes)
- RandomBrightnessContrast (±0.2)
- HueSaturationValue
- Normalization (CIFAR-100 mean/std)

### Test Augmentations

- Normalization only

## Output Files

After training, the following files are created in the checkpoint directory:

- `best_model.pth` - Model with highest test accuracy
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
from model import WideResNet

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth')

# Create model
model = WideResNet(depth=28, widen_factor=10, num_classes=100)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Test accuracy: {checkpoint['test_accuracy']:.2f}%")
```

## HuggingFace Integration

To enable automatic model upload to HuggingFace Hub:

1. **Create a HuggingFace account** at https://huggingface.co

2. **Create an access token**:
   - Go to Settings → Access Tokens
   - Create a token with write permissions

3. **Create a model repository** (optional, auto-created if not exists):
   - Go to your profile → New Model
   - Name it (e.g., `cifar100-wideresnet`)

4. **Train with HuggingFace upload**:
```bash
python train.py \
  --hf-token YOUR_TOKEN_HERE \
  --hf-repo your-username/cifar100-wideresnet
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
python train.py --batch-size 256 --num-workers 4
```

### For Apple Silicon
```bash
python train.py --batch-size 128 --num-workers 2
```

### For CPU Training
```bash
python train.py --batch-size 64 --num-workers 2 --no-amp
```

### Reduce Memory Usage
```bash
python train.py --batch-size 128 --no-amp
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
- Ensure data directory has write permissions

## Expected Results

With default settings (WideResNet-28-10, 100 epochs):

- **Training Time** (NVIDIA RTX 3090): ~2-3 hours
- **Training Time** (Apple M1 Pro): ~4-6 hours
- **Expected Test Accuracy**: 70-75%
- **Best Published Accuracy**: ~78% (with longer training)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cifar100-training,
  author = {Your Name},
  title = {CIFAR-100 Training Framework with WideResNet},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/resnet50-imagenet-1k}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- WideResNet architecture: [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
- MixUp augmentation: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- CIFAR-100 dataset: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
