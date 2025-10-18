import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils


def get_device():
    """Detect and return the best available device (CUDA, MPS, or CPU).

    Priority: CUDA > MPS > CPU
    Also enables appropriate backend optimizations.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    return device


def imshow_batch(img_batch, mean=None, std=None):
    img = img_batch.clone().cpu()
    npimg = img.numpy()
    if mean is None:
        mean = np.array([0.5071, 0.4867, 0.4408]).reshape(3,1,1)
    if std is None:
        std = np.array([0.2675, 0.2565, 0.2761]).reshape(3,1,1)
    npimg = (npimg * std) + mean
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.axis('off')
    plt.show()
