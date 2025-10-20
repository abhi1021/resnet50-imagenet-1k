"""
Model registry and factory for managing different architectures.
"""
import torch.nn as nn
from torchvision.models import resnet50 as torchvision_resnet50
from .wideresnet import WideResNet
from .resnet50 import ResNet50


def _get_pytorch_resnet50(num_classes=100, pretrained=False):
    """
    Get PyTorch's official ResNet50 model with custom number of classes.

    This is optimized for ImageNet (224x224) and is memory-efficient.
    """
    model = torchvision_resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    return model


# Model registry
MODELS = {
    'wideresnet28-10': lambda num_classes=100: WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropRate=0.3),
    'wideresnet': lambda num_classes=100: WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropRate=0.3),
    'resnet50': lambda num_classes=100: ResNet50(num_classes=num_classes),
    'resnet50-pytorch': lambda num_classes=100, **kwargs: _get_pytorch_resnet50(num_classes=num_classes, **kwargs),
    # Future models can be added here
    # 'resnet18': lambda num_classes=100: ResNet18(num_classes=num_classes),
    # 'efficientnet': lambda num_classes=100: EfficientNet(num_classes=num_classes),
}


def get_model(name, num_classes=100, **kwargs):
    """
    Factory function to get model by name.

    Args:
        name: Model name ('wideresnet28-10', 'resnet50', etc.)
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        torch.nn.Module: Initialized model

    Example:
        >>> model = get_model('wideresnet28-10', num_classes=100)
        >>> model = get_model('resnet50', num_classes=10)
    """
    name = name.lower()
    if name not in MODELS:
        available = ', '.join(MODELS.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    model_fn = MODELS[name]
    return model_fn(num_classes=num_classes, **kwargs)


def list_models():
    """
    List all available models.

    Returns:
        list: List of model names
    """
    return list(MODELS.keys())


__all__ = ['get_model', 'list_models', 'WideResNet', 'ResNet50']
