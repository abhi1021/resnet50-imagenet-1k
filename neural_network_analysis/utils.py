import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.backends.cudnn.benchmark = True
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
