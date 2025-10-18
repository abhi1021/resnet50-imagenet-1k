import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A


def get_transforms(img_size=64):
    train_transforms = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def transform_wrapper(pil_img):
        image_np = np.array(pil_img)
        augmented = train_transforms(image=image_np)
        return augmented['image']

    test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return transform_wrapper, test_transforms


def get_dataloaders(train_dir, val_dir, batch_size=64, img_size=64, num_workers=0, pin_memory=False):
    transform_train, transform_test = get_transforms(img_size=img_size)

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transform_test)

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, **dataloader_args)
    val_loader = DataLoader(val_ds, **dataloader_args)

    return train_loader, val_loader, train_ds
