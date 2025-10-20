"""
PyTorch's official ResNet50 implementation wrapper.

This module provides a wrapper around torchvision's ResNet50 model,
which is the standard implementation from "Deep Residual Learning for Image Recognition"
(https://arxiv.org/abs/1512.03385).

Architecture designed for ImageNet (224x224 images):
- Initial 7x7 conv with stride 2
- Max pooling layer
- 4 stages with bottleneck blocks: [3, 4, 6, 3]
- Adaptive average pooling
- Fully connected classifier

Usage:
    model = ResNet50PyTorch(num_classes=1000)  # ImageNet
    model = ResNet50PyTorch(num_classes=100)   # CIFAR-100
"""
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50PyTorch(nn.Module):
    """
    Wrapper for PyTorch's official ResNet50 implementation.

    This uses torchvision.models.resnet50 without pretrained weights,
    allowing you to train from scratch on your dataset.

    Args:
        num_classes: Number of output classes (default: 1000)

    Example:
        >>> model = ResNet50PyTorch(num_classes=1000)
        >>> model = ResNet50PyTorch(num_classes=10)  # Custom dataset
    """

    def __init__(self, num_classes=1000):
        super(ResNet50PyTorch, self).__init__()

        # Load ResNet50 architecture without pretrained weights
        # Using weights=None ensures we start from scratch
        self.model = resnet50(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
