# ImageNet Training Framework

A modular PyTorch training framework for ImageNet-1K with ResNet models, advanced augmentation, and mixed precision training.

## Setup Development Environment

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/yourusername/resnet50-imagenet-1k.git
cd resnet50-imagenet-1k

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

**For ImageNet-1K:**
- Download from https://image-net.org/download.php
- Extract and organize into train/val folders with class subfolders
- Expected structure: `/path/to/imagenet/{train,val}/n01440764/...`

**For ImageNette (10-class subset for local testing):**
```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf imagenette2-160.tgz
```

### 3. Run Training

```bash
# Quick test with ImageNette
python train.py --model resnet50-pytorch --dataset imagenet --data-dir ./imagenette2-160 --num-classes 10 --epochs 20 --batch-size 128

# Full ImageNet training
python train.py --model resnet50-pytorch --dataset imagenet --data-dir /path/to/imagenet --epochs 90 --batch-size 256
```

## Command-Line Flags

### Model & Dataset
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `wideresnet28-10` | Model architecture (`resnet50`, `resnet50-pytorch`, `wideresnet28-10`) |
| `--dataset` | str | `cifar100` | Dataset (`imagenet`, `cifar100`) |
| `--data-dir` | str | `../data` | Dataset directory path |
| `--num-classes` | int | `None` | Number of classes (e.g., 10 for ImageNette, auto-detected if not specified) |

### Training Parameters
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--epochs` | int | `100` | Number of training epochs |
| `--batch-size` | int | `256` | Training batch size |
| `--device` | str | `auto` | Device: `cuda`, `mps`, `cpu` (auto-detects if not specified) |

### Optimizer & Scheduler
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--optimizer` | str | `sgd` | Optimizer (`sgd`, `adam`, `adamw`) |
| `--scheduler` | str | `cosine` | LR scheduler (`cosine`, `onecycle`) |
| `--config` | str | `./config.json` | Path to config file with scheduler settings |

### Augmentation & Regularization
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--augmentation` | str | `strong` | Augmentation strength (`none`, `weak`, `strong`) |
| `--no-mixup` | flag | `False` | Disable MixUp augmentation |
| `--mixup-alpha` | float | `0.2` | MixUp interpolation parameter |
| `--label-smoothing` | float | `0.1` | Label smoothing factor |

### Training Features
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--no-amp` | flag | `False` | Disable automatic mixed precision (FP16) |
| `--gradient-clip` | float | `1.0` | Gradient clipping max norm |
| `--lr-finder` | flag | `False` | Run LR Finder before training (finds optimal learning rates) |

### Early Stopping
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--target-accuracy` | float | `None` | Target accuracy to achieve (e.g., `74.0` for 74%) |
| `--enable-target-early-stopping` | flag | `False` | Stop training when target accuracy is reached |

**Note:** Both `--target-accuracy` and `--enable-target-early-stopping` must be specified together. Without these flags, training runs for full epoch count with only patience-based early stopping (15 epochs no improvement).

### Checkpoint Management
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--resume-from` | str | `None` | Resume training from checkpoint directory or file path |
| `--keep-last-n-checkpoints` | int | `5` | Number of recent epoch checkpoints to keep (-1=all, 0=only breakpoints) |

### Visualization
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--visualize-samples` | flag | `False` | Visualize sample images from dataset before training |

### HuggingFace Integration
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hf-token` | str | `None` | HuggingFace API token for model upload |
| `--hf-repo` | str | `None` | HuggingFace repository ID (e.g., `username/model-name`) |

## Example Commands

### 1. Quick Test with ImageNette (Local)
```bash
# ResNet50-PyTorch on ImageNette (10 classes)
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir ./imagenette2-160 \
  --num-classes 10 \
  --epochs 20 \
  --batch-size 128 \
  --scheduler onecycle
```

### 2. ImageNette with LR Finder
```bash
# Automatically find optimal learning rates before training
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir ./imagenette2-160 \
  --num-classes 10 \
  --epochs 20 \
  --batch-size 128 \
  --scheduler onecycle \
  --lr-finder
```

### 3. Full ImageNet-1K Training (GPU)
```bash
# Standard ResNet50 on ImageNet-1K (1000 classes)
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --scheduler cosine \
  --augmentation strong
```

### 4. ImageNet with Target Accuracy Early Stopping
```bash
# Stop training when reaching 75% accuracy
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --target-accuracy 75.0 \
  --enable-target-early-stopping
```

### 5. ImageNet with Custom Augmentation
```bash
# Weak augmentation, no MixUp
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --augmentation weak \
  --no-mixup \
  --label-smoothing 0.0
```

### 6. ImageNet with OneCycle + LR Finder
```bash
# Use OneCycle scheduler with automatic LR finding
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --lr-finder
```

### 7. Apple Silicon (MPS) Optimized
```bash
# Optimized settings for M1/M2/M3 Macs
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir ./imagenette2-160 \
  --num-classes 10 \
  --epochs 20 \
  --batch-size 128 \
  --device mps
```

### 8. CPU Training (Lower Batch Size)
```bash
# For systems without GPU
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir ./imagenette2-160 \
  --num-classes 10 \
  --epochs 20 \
  --batch-size 64 \
  --no-amp \
  --device cpu
```

### 9. ImageNet with HuggingFace Upload
```bash
# Train and auto-upload to HuggingFace Hub
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --hf-token YOUR_HF_TOKEN \
  --hf-repo username/imagenet-resnet50
```

### 10. Minimal Training (Fast Testing)
```bash
# Minimal settings for quick testing/debugging
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir ./imagenette2-160 \
  --num-classes 10 \
  --epochs 5 \
  --batch-size 64 \
  --augmentation none \
  --no-mixup \
  --no-amp
```

### 11. ImageNet-1K with Resume + LR Finder
```bash
# Start fresh training with LR finder (first run)
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --lr-finder \
  --resume-from ./checkpoint_1

# Resume from checkpoint (subsequent runs)
# Note: LR finder is skipped when resuming - uses saved scheduler state
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --lr-finder \
  --resume-from ./checkpoint_1
```

## Available Models

- `resnet50` - Custom ResNet50 implementation (23.5M params)
- `resnet50-pytorch` - PyTorch ResNet50 (25.6M params, recommended for ImageNet)
- `wideresnet28-10` - WideResNet-28-10 (36.5M params, good for CIFAR)

## Output Files

Training creates a `checkpoint_N/` directory with:
- `best_model.pth` - Best model checkpoint
- `metrics.json` - Training history
- `training_curves.png` - Loss/accuracy plots
- `lr_finder_plot.png` - LR Finder results (if `--lr-finder` used)
- `config.json` - Training configuration
- `README.md` - Auto-generated model card

## Troubleshooting

**Out of Memory:**
- Reduce `--batch-size` (try 128, 64, 32)
- Add `--no-amp` to disable mixed precision
- Use a smaller model

**Slow on Apple Silicon:**
- Use `--batch-size 128` and `--device mps`
- Apple MPS works best with batch sizes 128-256

**Dataset Not Found:**
- Verify `--data-dir` path exists
- Check directory structure has `train/` and `val/` folders
- For ImageNette, ensure `--num-classes 10` is specified
