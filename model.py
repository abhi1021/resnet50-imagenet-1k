import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np

# ------------------------------
# Legacy CNN Model (CIFAR-10)
# ------------------------------
dropout_value = 0.05


class Net(nn.Module):
    """Legacy CNN model for CIFAR-10 (backward compatibility)."""

    def __init__(self):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1 (C1) — RF: 9
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0, bias=False),  # 32x32 → 30x30, RF=3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(32, 64, kernel_size=3, padding=0, bias=False),  # 30x30 → 28x28, RF=5
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, kernel_size=3, padding=0, dilation=2, bias=False),  # 28x28 → 24x24, RF=9
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 2 (C2) — RF: 13
        self.c2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False),  # 24x24 → 24x24, RF=9
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),  # Depthwise, RF=11
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False),  # Pointwise, RF=11
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=False),  # 24x24 → 22x22, RF=13
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 3 (C3) — RF: 21
        self.c3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=2, bias=False),  # 22x22 → 10x10, RF=17
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False),  # 10x10 → 10x10, RF=17
            nn.AvgPool2d(2, 2)  # 10x10 → 5x5, RF=21
        )

        # CONVOLUTION BLOCK 4 (C4) — RF: 44
        self.c4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),  # 5x5 → 5x5, RF=25
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  # 5x5 → 5x5, RF=29
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.AvgPool2d(kernel_size=5)  # 5x5 → 1x1, RF=44
        )

        # OUTPUT BLOCK (O) — 1x1x10
        self.output = nn.Conv2d(64, 10, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.c1(x)       # → 24x24x64
        x = self.c2(x)       # → 22x22x64
        x = self.c3(x)       # → 5x5x32
        x = self.c4(x)       # → 1x1x64
        x = self.output(x)   # → 1x1x10
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# ------------------------------
# WideResNet-28-10 Architecture
# ------------------------------
class BasicBlock(nn.Module):
    """Basic residual block for WideResNet."""

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.equalInOut = in_planes == out_planes
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropRate = dropRate
        self.shortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, 1, stride=stride, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        out = self.relu2(self.bn2(out))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = self.conv2(out)
        return out + (x if self.equalInOut else self.shortcut(x))


class NetworkBlock(nn.Module):
    """Network block consisting of multiple BasicBlocks."""

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet-28-10 for CIFAR-100.

    Args:
        depth: Total number of layers (default: 28)
        num_classes: Number of output classes (default: 100 for CIFAR-100)
        widen_factor: Width multiplier (default: 10)
        dropRate: Dropout rate (default: 0.3)
    """

    def __init__(self, depth=28, num_classes=100, widen_factor=10, dropRate=0.3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


# ------------------------------
# MixUp Data Augmentation
# ------------------------------
def mixup_data(x, y, alpha=0.2, device=None):
    """
    Apply MixUp augmentation to inputs and targets.

    Args:
        x: Input batch
        y: Target batch
        alpha: MixUp interpolation strength
        device: Device to use (torch.device or str). If None, uses x's device.

    Returns:
        tuple: (mixed_x, y_a, y_b, lambda) where lambda is the mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)

    # Use input tensor's device if not specified
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ------------------------------
# Learning Rate Warmup Scheduler
# ------------------------------
class WarmupScheduler:
    """
    Learning rate warmup scheduler.

    Gradually increases learning rate from initial_lr to target_lr over warmup_steps.
    """

    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_step = 0

    def step(self):
        """Update learning rate for one step."""
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1

    def is_warmup(self):
        """Check if still in warmup phase."""
        return self.current_step < self.warmup_steps


# ------------------------------
# Optimizer and Scheduler Configuration
# ------------------------------
def get_optimizer(model, model_name='wideresnet', lr=0.01, momentum=0.9, weight_decay=1e-3):
    """
    Get optimizer for the model.

    Args:
        model: Model to optimize
        model_name: Name of the model ('wideresnet' or 'net')
        lr: Initial learning rate
        momentum: SGD momentum
        weight_decay: Weight decay for regularization

    Returns:
        optimizer: Configured optimizer
    """
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_scheduler(optimizer, train_loader, scheduler_type='cosine', epochs=100):
    """
    Get learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        train_loader: Training data loader
        scheduler_type: Type of scheduler ('onecycle' or 'cosine')
        epochs: Total number of epochs

    Returns:
        scheduler: Configured scheduler
    """
    if scheduler_type == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=0.05,
            steps_per_epoch=len(train_loader),
            epochs=epochs
        )
    elif scheduler_type == 'cosine':
        # Cosine annealing with warm restarts (T_0 = 25 epochs per cycle)
        return CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-4)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ------------------------------
# Model Factory Function
# ------------------------------
def get_model(model_name='wideresnet', num_classes=100, **kwargs):
    """
    Factory function to get model by name.

    Args:
        model_name: Name of model ('wideresnet' or 'net')
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        model: Instantiated model
    """
    if model_name.lower() == 'wideresnet':
        return WideResNet(num_classes=num_classes, **kwargs)
    elif model_name.lower() == 'net':
        return Net()
    else:
        raise ValueError(f"Unknown model: {model_name}")
