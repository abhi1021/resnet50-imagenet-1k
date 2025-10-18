from __future__ import print_function
from torchvision import datasets, transforms, utils, models
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from torch.distributions.beta import Beta
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import notebook_login
from huggingface_hub import HfApi
import os, json

# Define your Hugging Face username and the desired model name
# Make sure 'your-username' is replaced with your actual Hugging Face username
hf_username = "AmolDuse"
model_name = "era-v4-resnet-50-imagenet-1k" # Choose a descriptive name
#notebook_login()

# Create a directory to save the model files
model_dir = f"./{model_name}"
os.makedirs(model_dir, exist_ok=True)

# Train Phase transformations

train_transforms = A.Compose([
    A.RandomResizedCrop(size=(64, 64), p=1.0),  # Then crop
    A.HorizontalFlip(p=0.5),    
    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def transform_wrapper(pil_img):
    """
    Wraps the Albumentations pipeline for torchvision datasets.
    
    1. Converts PIL Image -> NumPy Array (Albumentations input format).
    2. Runs the Albumentations pipeline (using the 'image=' named argument).
    3. Returns the augmented PyTorch Tensor.
    """
    # Convert PIL Image to NumPy array (H x W x C)
    image_np = np.array(pil_img)
    
    # Run the Albumentations pipeline, passing the image as a named argument
    augmented = train_transforms(image=image_np)
    
    # Return the PyTorch Tensor
    return augmented['image']

# Test Phase transformations
test_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Dataset and Creating Train/Test Split
# train = datasets.ImageFolder('/content/tiny-imagenet-200/train', transform=transform_wrapper)
# test = datasets.ImageFolder('/content/tiny-imagenet-200/val', transform=test_transforms)

train = datasets.ImageFolder('D:/learn-ai/assignment-9/resnet50-imagenet-1k/tiny-imagenet-200/train', transform=transform_wrapper)
test = datasets.ImageFolder('D:/learn-ai/assignment-9/resnet50-imagenet-1k/tiny-imagenet-200/val', transform=test_transforms)

# Dataloader Arguments & Test/Train Dataloaders
SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)


dataiter = iter(train_loader)
images, labels = next(dataiter)

print(images.shape)
print(labels.shape)

# Let's visualize some of the images

# Function to show an image
def imshow(img):

    # img: torch.Tensor (C,H,W) normalized with CIFAR-100 mean/std
    img = img.clone().cpu()
    npimg = img.numpy()
    mean = np.array([0.5071, 0.4867, 0.4408]).reshape(3,1,1)
    std = np.array([0.2675, 0.2565, 0.2761]).reshape(3,1,1)
    npimg = (npimg * std) + mean
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()
# Define class names for CIFAR-100 (use fine_label names from the dataset)
# torchvision's CIFAR100 stores class names in the dataset's 'classes' attribute
classes = train.classes if hasattr(train, 'classes') else []

# Show images from the batch
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
imshow(utils.make_grid(images[:4]))

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
if use_cuda:
    torch.backends.cudnn.benchmark = True

# Create model once and ensure num_classes=100
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 200) 
model = model.to(device) 
summary(model, input_size=(3, 64, 64))

# Training and Testing
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []
train_losses_epoch = []

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    batch_losses = []
    # CutMix params
    cutmix_prob = 0.5
    cutmix_alpha = 1.0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)
        
        # Init
        optimizer.zero_grad()       
        
        # Predict
        y_pred = model(data)
        # Calculate loss (CrossEntropyLoss accepts raw logits)
        loss = F.cross_entropy(y_pred, target)
        batch_losses.append(loss.item())
        train_losses.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Step the scheduler per batch if provided (OneCycleLR steps every batch)
        if scheduler is not None:
            scheduler.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    # epoch-level metrics
    epoch_loss = float(np.mean(batch_losses)) if len(batch_losses) > 0 else 0.0
    epoch_acc = 100. * correct / processed if processed > 0 else 0.0
    # record epoch-level loss
    train_losses_epoch.append(epoch_loss)
    return epoch_loss, epoch_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    epoch_acc = 100. * correct / len(test_loader.dataset)
    return test_loss, epoch_acc

# Switch to SGD + momentum (recommended baseline for ResNets)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
EPOCHS = 30

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
)

best_acc = 0.0

for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch, scheduler=scheduler)
    test_loss, test_accuracy = test(model, device, test_loader)
    print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Train acc {train_accuracy:.2f}%, Test loss {test_loss:.4f}, Test acc {test_accuracy:.2f}%")

    # Save best model checkpoint by validation accuracy
    try:
        # find best epoch and save (we only have last epoch metrics in memory here)
        # Instead, save if current test_accuracy is best
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            #torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'test_acc': test_accuracy}, os.path.join(model_dir, "pytorch_model.pt"))
    
            if best_acc > 50.0:
              # Save the configuration if you have any specific to your model
              # For a simple custom model, this might not be strictly necessary but is good practice
              # Example: Save a dictionary with num_classes
              config = {"num_classes": 100}
              with open(os.path.join(model_dir, "config.json"), "w") as f:
                  json.dump(config, f)

              repo_id = f"{hf_username}/{model_name}"
              # After training, save and push:
              # Save the model using save_pretrained which handles model saving in the correct format for the hub
              # model.save_pretrained(model_name) # This saves as model.safetensors by default
              # Explicitly add the pytorch_model.bin file before pushing
              api = HfApi()
            #   api.upload_file(
            #       path_or_fileobj=os.path.join(model_dir, "pytorch_model.pt"),
            #       path_in_repo="pytorch_model.pt",
            #       repo_id=repo_id,
            #       repo_type="model",
            #   )

              # You can still push other files saved with save_pretrained or manually
              # model.push_to_hub(f"{repo_id}") # This will push other changes like config.json or README if saved

              print(f"Model state dictionary saved locally to {os.path.join(model_dir, 'pytorch_model.pt')} and uploaded to https://huggingface.co/{repo_id}")
    except Exception:
      pass

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,2,figsize=(15,10))
# Plot epoch-level training loss if available, otherwise per-batch
if len(train_losses_epoch) > 0:
    axs[0, 0].plot(train_losses_epoch)
else:
    axs[0, 0].plot(train_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")

# Try to run Grad-CAM on the best checkpoint if the helper is available
try:
    from neural_network_analysis.gradcam_pytorch import show_gradcam_batch
    # load best checkpoint if exists
    import os
    if os.path.exists('best.pth'):
        checkpoint = torch.load('best.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best checkpoint with test_acc={checkpoint.get('test_acc')}")
    else:
        print('No best.pth checkpoint found — running Grad-CAM on current model')
    show_gradcam_batch(model, device, test_loader, classes=classes, n=6)
except Exception as e:
    print('Grad-CAM helper not available or failed:', e)