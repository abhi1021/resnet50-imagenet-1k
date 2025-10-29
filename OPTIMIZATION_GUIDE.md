# ResNet50 ImageNet-1K Training Optimization Guide

## Summary of Changes

This guide documents the optimizations applied to improve training convergence and reach 74%+ accuracy more efficiently.

### Changes Made

#### 1. Learning Rate Warmup (NEW Feature)
- **Added**: `WarmupScheduler` class in `training/scheduler.py`
- **Purpose**: Gradually increases learning rate during the first few epochs to stabilize training
- **Benefit**: Prevents early training instability and divergence with higher learning rates

#### 2. Optimized Hyperparameters in `config.json`

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `initial_lr` | 0.05 | 0.1 | 2x increase for faster convergence with warmup protection |
| `T_0` (cosine) | 30 | 100 | Match total training epochs for smooth decay |
| `warmup_epochs` | N/A | 5 | 5-epoch linear warmup before cosine schedule |
| `warmup_start_lr` | N/A | 0.0001 | Starting LR for warmup phase |

### How It Works

#### Learning Rate Schedule (100 epochs)

```
Epoch 1-5 (Warmup):
  LR increases linearly: 0.0001 → 0.1

Epoch 6-100 (Cosine Annealing):
  LR decreases via cosine: 0.1 → 0.00001
```

**Visual representation:**
```
    0.1 ┤     ╭─────────────────╮
        │    ╱                   ╲
        │   ╱                     ╲
  0.05  │  ╱                       ╲
        │ ╱                         ╲
        │╱                           ╲___
 0.0001 ┤                                ╲___
        └─────────────────────────────────────
        1   5   10      50        100
        └─┬─┘   └────────┬────────────┘
        Warmup    Cosine Annealing
```

## Usage Instructions

### Continue Your Current Training

Since you're already at epoch 8, you have two options:

#### Option A: Resume with New Settings (Recommended)
```bash
# Resume from your checkpoint with the new optimized config
python train.py \
  --model resnet50 \
  --dataset imagenet-1k \
  --data-dir ../data \
  --epochs 100 \
  --batch-size 256 \
  --scheduler cosine \
  --resume-from ./checkpoint_5 \
  --config ./config.json
```

**Note**: When resuming, the warmup will be skipped since you're past epoch 5. The new higher LR will apply gradually through the cosine schedule.

#### Option B: Start Fresh with New Settings
```bash
# Start new training from scratch with optimized settings
python train.py \
  --model resnet50 \
  --dataset imagenet-1k \
  --data-dir ../data \
  --epochs 100 \
  --batch-size 256 \
  --scheduler cosine \
  --config ./config.json
```

**Recommendation**: Continue with Option A (resume) since you already have 8 epochs of training completed. The optimizations will still help in the remaining epochs.

### What to Expect

With these optimizations on your current training (resuming from epoch 8):

| Metric | Current (Epoch 8) | Expected (with optimizations) |
|--------|-------------------|-------------------------------|
| Epoch 15 | ~55-58% | ~60-63% |
| Epoch 25 | ~63-66% | ~68-71% |
| Epoch 40 | ~69-72% | **~74%+ (TARGET)** |
| Epoch 80-100 | ~72-75% | ~76-77% |

### Monitor Training Progress

The learning rate schedule will be visible in your training output:
```
Epoch 9: 100%|██████| 5005/5005 [44:00<00:00, 1.89it/s, loss=X.XX, acc=XX.XX%, lr=0.099XXX]
                                                                              ^^^^^^^^
                                                                              Watch this LR
```

### Key Training Curves to Monitor

1. **Learning Rate Curve**: Should show smooth cosine decay
2. **Train-Test Gap**: Should remain around -5% to -12% (negative is normal)
3. **Loss Curves**: Both should decrease smoothly without sudden jumps
4. **Accuracy**: Test accuracy should increase steadily

## Advanced Configuration

### Adjust Warmup Settings

If you want to modify warmup behavior, edit `config.json`:

```json
{
  "scheduler": {
    "cosine": {
      "warmup_epochs": 5,        // Increase to 7-10 for more gradual warmup
      "warmup_start_lr": 0.0001  // Lower to 0.00001 for even gentler start
    }
  }
}
```

### Recommendations by Training Stage

**If training is unstable early on:**
- Increase `warmup_epochs` to 7-10
- Decrease `warmup_start_lr` to 0.00001
- Decrease `initial_lr` to 0.08

**If convergence is too slow:**
- Increase `initial_lr` to 0.12-0.15
- Decrease `warmup_epochs` to 3
- Ensure strong augmentation is enabled

**If overfitting (train >> test accuracy):**
- Increase `weight_decay` from 0.0001 to 0.0002
- Increase `mixup_alpha` from 0.2 to 0.3
- Increase `label_smoothing` from 0.1 to 0.15

## Technical Details

### WarmupScheduler Implementation

The `WarmupScheduler` class wraps any PyTorch scheduler with a linear warmup phase:

```python
# During warmup (epochs 1-5):
lr = warmup_start_lr + (initial_lr - warmup_start_lr) * (current_epoch / warmup_epochs)

# After warmup (epochs 6-100):
lr = cosine_annealing(initial_lr, eta_min, current_epoch - warmup_epochs)
```

### Checkpoint Compatibility

The warmup scheduler state is saved in checkpoints:
```python
{
  'current_epoch': int,
  'finished_warmup': bool,
  'scheduler_state': {...}
}
```

When resuming, the scheduler automatically detects whether warmup is complete.

## Troubleshooting

### Issue: Training crashes with "RuntimeError: CUDA out of memory"
**Solution**: The higher learning rate doesn't affect memory. This is likely a different issue.
- Check batch size (should be 256 or adjust based on GPU)
- Verify mixed precision is enabled: `--no-amp` should NOT be used

### Issue: Loss suddenly spikes
**Solution**:
- Increase warmup period to 7-10 epochs
- Reduce `initial_lr` to 0.08
- Check gradient clipping is enabled (default: 1.0)

### Issue: Accuracy plateaus below 74%
**Solution**:
- Run for full 100 epochs
- Verify strong augmentation is enabled: `--augmentation strong`
- Check MixUp is enabled (should be default)
- Consider adjusting weight decay or other regularization

## References

- **Learning Rate Warmup**: Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)
- **ResNet ImageNet Training**: He et al. "Deep Residual Learning for Image Recognition" (2015)
- **Cosine Annealing**: Loshchilov & Hutter "SGDR: Stochastic Gradient Descent with Warm Restarts" (2016)

## Questions?

If you encounter issues or need further optimization, check:
1. Training curves in `./checkpoint_5/training_curves.png`
2. Metrics history in `./checkpoint_5/metrics.json`
3. Configuration in `./checkpoint_5/config.json`
