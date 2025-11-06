# ImageNet Training Framework

A modular PyTorch training framework for ImageNet-1K with ResNet models, advanced augmentation, and mixed precision training.

## Training Results

**Best Accuracy Achieved: 73.84%** on ImageNet-1K (epoch 90)

### Model Performance
- **Architecture**: ResNet50-PyTorch (25.6M parameters)
- **Dataset**: ImageNet-1K (1000 classes)
- **Best Test Accuracy**: 73.84% (achieved at epoch 90)
- **Total Training**: 92 epochs across 3 learning rate cycles
- **Final Metrics (Epoch 92)**:
  - Train Accuracy: 42.78% | Train Loss: 3.665
  - Test Accuracy: 52.61% | Test Loss: 2.166

### Training Configuration
- **Batch Size**: 256
- **Optimizer**: SGD (momentum=0.9, weight_decay=1e-3)
- **Scheduler**: Cosine with restarts (3 cycles)
- **Augmentation**: Strong (HorizontalFlip, ShiftScaleRotate, Cutout, ColorJitter)
- **MixUp**: α=0.2
- **Label Smoothing**: 0.1
- **Mixed Precision**: Enabled (FP16)
- **Gradient Clipping**: 1.0

### Training Progression
The model was trained with cosine learning rate scheduling across 3 complete cycles (epochs 1-30, 31-60, 61-92), achieving peak performance at epoch 90 during the final cycle.


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

### Multi-GPU Training
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--multi-gpu` | flag | `False` | Enable multi-process DistributedDataParallel (launch with torchrun) |
| `--sync-bn` | flag | `False` | Convert BatchNorm to SyncBatchNorm before wrapping with DDP |

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
| `--hf-token` | str | `None` | HuggingFace API token (required for gated datasets like ImageNet-1K and for model upload) |
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

### 8. Multi-GPU Training (DistributedDataParallel)
```bash
# Train on 4 GPUs with torchrun (distributes batch across GPUs)
torchrun --nproc_per_node=4 train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --multi-gpu \
  --scheduler cosine

# Multi-GPU with SyncBatchNorm (recommended for better sync across GPUs)
torchrun --nproc_per_node=4 train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --multi-gpu \
  --sync-bn \
  --scheduler cosine

# Multi-GPU with OneCycle scheduler and LR Finder
torchrun --nproc_per_node=2 train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir ./imagenette2-160 \
  --num-classes 10 \
  --epochs 20 \
  --batch-size 256 \
  --multi-gpu \
  --sync-bn \
  --scheduler onecycle \
  --lr-finder

# Alternative: Using python -m torch.distributed.launch (older PyTorch versions)
python -m torch.distributed.launch --nproc_per_node=4 train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir /path/to/imagenet \
  --epochs 90 \
  --batch-size 256 \
  --multi-gpu \
  --sync-bn
```

**Important Notes:**
- `--batch-size` is the **total batch size** - it's automatically divided across GPUs
- With 4 GPUs and `--batch-size 256`, each GPU processes 64 samples per batch
- Use `torchrun` (recommended for PyTorch 1.9+) or `python -m torch.distributed.launch` (older versions)
- `--nproc_per_node` should match the number of GPUs you want to use
- Only rank 0 process saves checkpoints and prints logs to avoid duplicates
- `--sync-bn` synchronizes BatchNorm statistics across GPUs for better training stability

### 9. CPU Training (Lower Batch Size)
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

### 10. ImageNet with HuggingFace Upload
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

### 11. Minimal Training (Fast Testing)
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

### 12. ImageNet-1K (HuggingFace) with Resume + LR Finder
```bash
# IMPORTANT: ImageNet-1K is a gated dataset requiring authentication
# First, request access at: https://huggingface.co/datasets/ILSVRC/imagenet-1k
# Then get your token at: https://huggingface.co/settings/tokens

# Start fresh training with LR finder (first run)
# Loads ImageNet-1K from HuggingFace (ILSVRC/imagenet-1k)
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir ./hf_cache \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --lr-finder \
  --resume-from ./checkpoint_1 \
  --hf-token YOUR_HF_TOKEN_HERE

# Resume from checkpoint (subsequent runs)
# Note: LR finder is skipped when resuming - uses saved scheduler state
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir ./hf_cache \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --lr-finder \
  --resume-from ./checkpoint_1 \
  --hf-token YOUR_HF_TOKEN_HERE

# Alternative: Authenticate once via CLI (no --hf-token needed)
# huggingface-cli login
```

## Available Models

- `resnet50` - Custom ResNet50 implementation (23.5M params)
- `resnet50-pytorch` - PyTorch ResNet50 (25.6M params, recommended for ImageNet)
- `wideresnet28-10` - WideResNet-28-10 (36.5M params, good for CIFAR)

## Output Files

Training creates a `checkpoint_N/` directory with:
- `best_model.pth` - Best model checkpoint
- `training_state_epochN.pth` - Full training state for each epoch (for resumption)
- `metrics.json` - Training history
- `training_curves.png` - Loss/accuracy plots
- `lr_finder_plot.png` - LR Finder results (if `--lr-finder` used)
- `lr_finder_results.json` - Saved LR finder results (automatically reused on resume)
- `config.json` - Training configuration
- `README.md` - Auto-generated model card

## Resume Training

The framework supports robust checkpoint resumption with automatic LR finder result reuse.

### Basic Resume Usage

```bash
# Resume from a checkpoint directory (automatically finds latest epoch)
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir ./hf_cache \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --resume-from ./checkpoint_1 \
  --hf-token YOUR_TOKEN

# Resume from a specific checkpoint file
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir ./hf_cache \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --resume-from ./checkpoint_1/training_state_epoch5.pth \
  --hf-token YOUR_TOKEN
```

### Handling Training Interruptions

**Scenario 1: Training interrupted during epoch 1 (before first checkpoint saved)**

If training is interrupted during epoch 1, no `training_state_epoch1.pth` exists yet. The framework handles this gracefully:

```bash
# First run - gets interrupted during epoch 1
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir ./hf_cache \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --lr-finder \
  --resume-from ./checkpoint_1 \
  --hf-token YOUR_TOKEN

# Training runs LR finder, saves results to lr_finder_results.json
# Training starts epoch 1 but gets interrupted
# NO training_state_epoch1.pth created yet

# Resume - automatically handles missing checkpoint
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir ./hf_cache \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --lr-finder \
  --resume-from ./checkpoint_1 \
  --hf-token YOUR_TOKEN

# Output:
# ⚠️  WARNING: No training state checkpoints found in ./checkpoint_1
#    Starting training from scratch, but will use this directory for checkpoints.
# ✓ Found saved LR finder results from previous run
#    Will reuse these results instead of re-running LR finder
```

**What happens:**
1. Shows warning about missing checkpoints (expected behavior)
2. Starts training from epoch 1 (fresh start)
3. **Automatically loads and reuses saved LR finder results** from `lr_finder_results.json`
4. Skips re-running expensive LR range test
5. Applies saved learning rates to scheduler
6. Continues training normally

**Scenario 2: Normal resume from completed epoch**

```bash
# Resume from epoch 5 (training_state_epoch5.pth exists)
python train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir ./hf_cache \
  --epochs 90 \
  --batch-size 256 \
  --scheduler onecycle \
  --lr-finder \
  --resume-from ./checkpoint_1 \
  --hf-token YOUR_TOKEN

# Output:
# 🔍 Found latest checkpoint at epoch 5: ./checkpoint_1/training_state_epoch5.pth
# ✓ Loaded training state from: ./checkpoint_1/training_state_epoch5.pth
# ✓ Restored model state
# ✓ Restored optimizer state
# ✓ Restored scheduler state
# ℹ SKIPPING LR FINDER
#    Resuming from checkpoint - using saved scheduler state
# RESUMING TRAINING FROM EPOCH 6
```

**What happens:**
1. Finds and loads latest checkpoint (`training_state_epoch5.pth`)
2. Restores complete training state (model, optimizer, scheduler, metrics, RNG)
3. Skips LR finder (scheduler already has correct learning rates from checkpoint)
4. Continues training from epoch 6

### LR Finder State Persistence

When `--lr-finder` is used, results are automatically saved to `lr_finder_results.json`:

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "scheduler_type": "onecycle",
  "selection_method": "steepest_gradient",
  "suggested_lrs": {
    "max_lr": 0.123456,
    "base_lr": 0.012345
  },
  "config_update": {
    "max_lr": 0.123456,
    "base_lr": 0.012345,
    "div_factor": 10.0
  }
}
```

**Benefits:**
- No need to re-run expensive LR range test on resume
- Consistent learning rates across training interruptions
- Results saved even if training crashes during epoch 1

### Resume Best Practices

1. **Always use the same critical parameters** when resuming:
   - `--model`, `--dataset`, `--data-dir`, `--num-classes`, `--batch-size`
   - The script validates these and will error if they don't match

2. **Checkpoint cleanup** is automatic:
   - Keeps last N epoch checkpoints (default: 5, configurable via `--keep-last-n-checkpoints`)
   - Breakpoint epochs (10, 20, 25, 30, 40, 50, 60, 75, 90) are kept forever
   - Best model checkpoint is always preserved

3. **LR finder behavior**:
   - Only runs when starting fresh (epoch 1)
   - Automatically reused if saved results exist
   - Skipped when resuming from actual checkpoint (uses saved scheduler state)

## Troubleshooting

**No training state checkpoints found (Resume Warning):**
```
⚠️  WARNING: No training state checkpoints found in ./checkpoint_1
   Looking for files matching pattern: training_state_epoch*.pth
   This can happen if training was interrupted before first epoch completed.
   Starting training from scratch, but will use this directory for checkpoints.
```
**What this means:**
- This is **expected behavior**, not an error
- Happens when training was interrupted during epoch 1 (before first checkpoint was saved)
- Training will restart from epoch 1 but reuse the checkpoint directory
- If LR finder was run previously, results will be automatically loaded and reused
- No need to worry - training will continue normally

**HuggingFace Authentication Error (ImageNet-1K):**
```
Dataset 'ILSVRC/imagenet-1k' is a gated dataset on the Hub. You must be authenticated to access it.
```
**Solution:**
1. Request access at: https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. Wait for access approval (usually instant if you have ImageNet license)
3. Get your token at: https://huggingface.co/settings/tokens
4. Use the token:
   - **Option 1:** Pass via CLI: `--hf-token YOUR_TOKEN_HERE`
   - **Option 2:** Login once: `huggingface-cli login` (then no need for `--hf-token`)

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
