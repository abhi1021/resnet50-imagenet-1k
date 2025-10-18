import torch.nn as nn
from torchvision import models


def build_resnet(model_name='resnet18', num_classes=200, pretrained=False):
    """Return a ResNet model with modified final layer."""
    model_map = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
    }

    if model_name not in model_map:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = model_map[model_name](weights=None if not pretrained else 'IMAGENET1K_V1')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
