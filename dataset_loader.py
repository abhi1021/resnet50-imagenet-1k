from datasets import load_dataset
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 1. Load from disk
# dataset_path = "/mnt/imagenet/dataset/ILSVRC___imagenet-1k/default/0.0.0/49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
# dataset_path = "/mnt/imagenet/dataset/ILSVRC___imagenet-1k"
# ds = load_from_disk(dataset_path)

# 1. Load from actual/cache
ds = load_dataset("ILSVRC/imagenet-1k", cache_dir="/mnt/imagenet/dataset")
train_ds = ds["train"]
val_ds = ds["validation"]

# 2. Define transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 3. Collate function
def collate_fn(batch):
    imgs = [transform(ex["image"]) for ex in batch]
    lbls = [ex["label"] for ex in batch]
    return torch.stack(imgs), torch.tensor(lbls)


# 4. DataLoader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn)

# Training loop (example)
# for images, labels in train_loader:
# images: [batch, 3, 224, 224], labels: [batch]
#    pass  # your training code here

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # Imagenette has 10 classes
model = model.to(device)

print(f"{device}: Looks ok till now!!!")

# Assuming you loaded your dataset like: train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")
# Get class names from dataset features
class_names = train_ds.features['label'].names  # Gets all 1000 ImageNet class names

print(f"Class names: {class_names[0:10]}")

# Get a batch of training data
batch_data, batch_label = next(iter(train_loader))

# Create figure with subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))
fig.suptitle('Sample Images from Training Batch', fontsize=16, y=0.995)

# Flatten axes array for easier iteration
axes = axes.flatten()

# Plot each image
for idx in range(12):
    img = batch_data[idx].clone()  # Clone to avoid modifying original

    # Handle both grayscale (1 channel) and RGB (3 channels) images
    if img.shape[0] == 1:
        # Grayscale image: squeeze the channel dimension
        img = img.squeeze(0).numpy()  # Shape: [H, W]
        axes[idx].imshow(img, cmap='gray')  # Use grayscale colormap
    else:
        # RGB image: convert from CHW to HWC format
        img = img.permute(1, 2, 0).numpy()  # Shape: [H, W, 3]
        img = np.clip(img, 0, 1)
        axes[idx].imshow(img)

    # Set title with class name from dataset
    label = batch_label[idx].item()
    class_name = class_names[label]
    axes[idx].set_title(f'{class_name}', fontsize=9, pad=5)

    # Remove axis ticks
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

    # Optional: Add thin border around images
    for spine in axes[idx].spines.values():
        spine.set_edgecolor('lightgray')
        spine.set_linewidth(0.5)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()