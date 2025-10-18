import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotille
import os
import json
from datetime import datetime

# Optional imports for model summary - gracefully handle if not installed
try:
    from torchinfo import summary as torchinfo_summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

try:
    from torchsummary import summary as torchsummary_summary
    TORCHSUMMARY_AVAILABLE = True
except ImportError:
    TORCHSUMMARY_AVAILABLE = False

if not TORCHINFO_AVAILABLE and not TORCHSUMMARY_AVAILABLE:
    print("⚠ Warning: Neither torchinfo nor torchsummary is installed.")
    print("   Install one with: pip install torchinfo  OR  pip install torchsummary")
elif not TORCHINFO_AVAILABLE:
    print("⚠ Using torchsummary for model summary (torchinfo not installed)")

# Import data loading utilities
from data import get_cifar100_loaders, get_imagenet_loaders

# Import model utilities
from model import (
    get_model, get_optimizer, get_scheduler,
    mixup_data, WarmupScheduler
)


def get_device():
    """
    Detect and return the best available device.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU

    Returns:
        tuple: (device, device_type) where device_type is 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print("⚠ Using CPU (no GPU detected)")

    return device, device_type


class ImageClassificationTrainer:
    """
    Trainer for image classification (CIFAR-100 and ImageNet) with advanced features:
    - Mixed precision training
    - MixUp augmentation
    - Label smoothing
    - Gradient clipping
    - Warmup + Cosine annealing scheduler
    - HuggingFace integration
    - Checkpoint management
    """

    def __init__(
        self,
        model_name='resnet50',
        dataset='imagenet',
        epochs=100,
        batch_size=256,
        data_dir='./data',
        num_workers=4,
        # Learning rate settings
        initial_lr=0.01,
        max_lr=0.1,
        min_lr=1e-4,
        warmup_epochs=5,
        scheduler_type='cosine',
        # Regularization
        weight_decay=1e-3,
        label_smoothing=0.1,
        gradient_clip=1.0,
        # Data augmentation
        use_mixup=True,
        mixup_alpha=0.2,
        # Training optimizations
        use_mixed_precision=True,
        # Checkpointing
        checkpoint_dir='./checkpoints',
        checkpoint_epochs=None,
        # HuggingFace
        hf_token=None,
        hf_repo=None,
        # Model-specific kwargs
        **model_kwargs
    ):
        """
        Initialize image classification trainer.

        Args:
            model_name: Name of model ('resnet18', 'resnet34', 'resnet50', 'wideresnet', 'net')
            dataset: Dataset to train on ('imagenet' or 'cifar100')
            epochs: Number of training epochs
            batch_size: Batch size
            data_dir: Directory for dataset (ImageNet root or CIFAR-100 data dir)
            num_workers: Number of data loading workers
            initial_lr: Initial learning rate (for warmup)
            max_lr: Maximum learning rate (after warmup)
            min_lr: Minimum learning rate (cosine annealing)
            warmup_epochs: Number of warmup epochs
            scheduler_type: 'cosine' or 'onecycle'
            weight_decay: Weight decay for regularization
            label_smoothing: Label smoothing factor
            gradient_clip: Gradient clipping max norm
            use_mixup: Whether to use MixUp augmentation
            mixup_alpha: MixUp interpolation strength
            use_mixed_precision: Whether to use FP16 training
            checkpoint_dir: Directory to save checkpoints
            checkpoint_epochs: List of epochs to save checkpoints (e.g., [10, 25, 50])
            hf_token: HuggingFace API token (optional)
            hf_repo: HuggingFace repository ID (optional)
            **model_kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.dataset = dataset.lower()
        self.epochs = epochs
        self.batch_size = batch_size
        self.device, self.device_type = get_device()

        # Learning rate settings (num_classes will be determined after loading dataset)
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler_type

        # Regularization
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.gradient_clip = gradient_clip

        # Data augmentation
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # Training optimizations
        self.use_mixed_precision = use_mixed_precision

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_epochs = checkpoint_epochs or []
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # HuggingFace integration
        self.hf_token = hf_token
        self.hf_repo = hf_repo
        self.hf_api = None
        if self.hf_token and self.hf_repo:
            self._setup_huggingface()

        # Setup data loaders first to determine number of classes
        # Pin memory only for CUDA (not beneficial for MPS)
        use_pin_memory = (self.device_type == 'cuda')

        if self.dataset == 'imagenet':
            print(f"\nLoading ImageNet dataset from: {data_dir}")
            self.train_loader, self.test_loader, self.train_dataset, self.test_dataset = \
                get_imagenet_loaders(
                    data_dir=data_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=use_pin_memory
                )
            # Determine number of classes from dataset
            self.num_classes = len(self.train_dataset.classes)
            print(f"Detected {self.num_classes} classes in dataset")
        elif self.dataset == 'cifar100':
            print(f"\nLoading CIFAR-100 dataset")
            self.train_loader, self.test_loader, self.train_dataset, self.test_dataset = \
                get_cifar100_loaders(
                    data_dir=data_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=use_pin_memory
                )
            self.num_classes = 100
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Choose 'imagenet' or 'cifar100'")

        # Initialize model with detected number of classes
        self.model = get_model(model_name, num_classes=self.num_classes, **model_kwargs).to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(
            self.model,
            model_name=model_name,
            lr=initial_lr,
            weight_decay=weight_decay
        )

        self.scheduler = get_scheduler(
            self.optimizer,
            self.train_loader,
            scheduler_type=scheduler_type,
            epochs=epochs
        )

        # Setup warmup scheduler
        self.warmup_scheduler = WarmupScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            initial_lr=initial_lr,
            target_lr=max_lr,
            steps_per_epoch=len(self.train_loader)
        )

        # Setup mixed precision scaler
        if self.use_mixed_precision:
            if self.device_type == 'cuda':
                self.scaler = torch.amp.GradScaler('cuda')
            elif self.device_type == 'mps':
                # MPS doesn't support GradScaler yet, use CPU scaler
                # Note: Mixed precision on MPS is handled differently
                self.scaler = torch.amp.GradScaler('cpu')
                print("⚠ MPS device detected: Mixed precision may have limited support")
            else:
                self.scaler = torch.amp.GradScaler('cpu')
        else:
            self.scaler = None

        # Initialize metric tracking
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.learning_rates = []
        self.best_test_acc = 0.0

    def _setup_huggingface(self):
        """Setup HuggingFace Hub integration."""
        try:
            from huggingface_hub import HfApi, create_repo

            self.hf_api = HfApi()
            # Create repository if it doesn't exist
            create_repo(
                repo_id=self.hf_repo,
                repo_type="model",
                exist_ok=True,
                token=self.hf_token
            )
            print(f"✓ HuggingFace repository ready: https://huggingface.co/{self.hf_repo}")
        except Exception as e:
            print(f"Warning: HuggingFace setup failed: {e}")
            self.hf_api = None

    def _upload_to_huggingface(self, file_path, path_in_repo, commit_message="Upload file"):
        """Upload a file to HuggingFace Hub."""
        if not self.hf_api or not self.hf_token:
            return

        try:
            self.hf_api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=self.hf_repo,
                repo_type="model",
                token=self.hf_token,
                commit_message=commit_message
            )
            print(f"✓ Uploaded: {path_in_repo}")
        except Exception as e:
            print(f"✗ Upload failed for {path_in_repo}: {e}")

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (average_loss, accuracy) for the epoch
        """
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        correct = 0
        processed = 0
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Mixed precision training
            if self.use_mixed_precision:
                with torch.amp.autocast(self.device_type):
                    if self.use_mixup:
                        # Apply MixUp
                        inputs, targets_a, targets_b, lam = mixup_data(
                            data, target, alpha=self.mixup_alpha, device=self.device
                        )
                        outputs = self.model(inputs)
                        loss = lam * F.cross_entropy(outputs, targets_a, label_smoothing=self.label_smoothing) + \
                               (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=self.label_smoothing)
                    else:
                        outputs = self.model(data)
                        loss = F.cross_entropy(outputs, target, label_smoothing=self.label_smoothing)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                if self.use_mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(
                        data, target, alpha=self.mixup_alpha, device=self.device
                    )
                    outputs = self.model(inputs)
                    loss = lam * F.cross_entropy(outputs, targets_a, label_smoothing=self.label_smoothing) + \
                           (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=self.label_smoothing)
                else:
                    outputs = self.model(data)
                    loss = F.cross_entropy(outputs, target, label_smoothing=self.label_smoothing)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.optimizer.step()

            # Update learning rate
            if self.warmup_scheduler.is_warmup():
                self.warmup_scheduler.step()
            else:
                if self.scheduler_type == 'cosine':
                    self.scheduler.step()
                # OneCycle scheduler is stepped per batch regardless

            # Calculate accuracy
            _, pred = outputs.max(1)
            if self.use_mixup:
                correct += lam * pred.eq(targets_a).sum().item() + (1 - lam) * pred.eq(targets_b).sum().item()
            else:
                correct += pred.eq(target).sum().item()
            processed += len(data)
            epoch_loss += loss.item()

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_description(
                f"Epoch {epoch} Loss={loss.item():.4f} Acc={100*correct/processed:.2f}% LR={current_lr:.6f}"
            )

        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100. * correct / processed
        return avg_loss, accuracy

    def test(self):
        """
        Test the model and return loss and accuracy.

        Returns:
            tuple: (test_loss, accuracy) for the test set
        """
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        acc = 100. * correct / len(self.test_loader.dataset)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(self.test_loader.dataset)} ({acc:.2f}%)\n"
        )
        return test_loss, acc

    def save_checkpoint(self, epoch, train_acc, test_acc, train_loss, test_loss,
                        checkpoint_name, is_best=False):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'best_test_acc': self.best_test_acc,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': self.model_name,
                'batch_size': self.batch_size,
                'mixup_alpha': self.mixup_alpha if self.use_mixup else None,
                'label_smoothing': self.label_smoothing,
                'weight_decay': self.weight_decay,
            }
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_name}")

        # Upload to HuggingFace if configured
        if self.hf_api:
            commit_msg = f"Epoch {epoch}: Test Acc {test_acc:.2f}%"
            if is_best:
                commit_msg = f"New best model! " + commit_msg
            self._upload_to_huggingface(checkpoint_path, checkpoint_name, commit_message=commit_msg)

        return checkpoint_path

    def create_model_card(self):
        """Create and upload README.md model card."""
        model_card = f"""---
tags:
- image-classification
- cifar100
- {self.model_name}
- pytorch
datasets:
- cifar100
metrics:
- accuracy
---

# CIFAR-100 {self.model_name.upper()}

## Model Description

{self.model_name.upper()} trained on CIFAR-100 dataset with advanced augmentation techniques.

### Training Configuration
- **Model**: {self.model_name}
- **Batch Size**: {self.batch_size}
- **Optimizer**: SGD (momentum=0.9, weight_decay={self.weight_decay})
- **Learning Rate**: Warmup ({self.initial_lr}→{self.max_lr}), Scheduler: {self.scheduler_type}
- **MixUp**: {'Enabled (alpha=' + str(self.mixup_alpha) + ')' if self.use_mixup else 'Disabled'}
- **Label Smoothing**: {self.label_smoothing}
- **Mixed Precision**: {'Enabled' if self.use_mixed_precision else 'Disabled'}
- **Gradient Clipping**: {self.gradient_clip}

### Performance
- **Best Test Accuracy**: {self.best_test_acc:.2f}%
- **Total Epochs Trained**: {len(self.train_accuracies)}

### Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
checkpoint_path = hf_hub_download(
    repo_id="{self.hf_repo}",
    filename="best_model.pth"
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load model (define model architecture first)
# ... instantiate model ...
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Dataset
- **Dataset**: CIFAR-100 (50,000 train, 10,000 test)
- **Classes**: 100
- **Image Size**: 32×32

### Files
- `best_model.pth` - Best performing model
- `final_model.pth` - Final epoch model
- `training_curves.png` - Training/test metrics visualization
- `metrics.json` - Complete training history

### License
MIT
"""

        readme_path = os.path.join(self.checkpoint_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(model_card)

        if self.hf_api:
            self._upload_to_huggingface(readme_path, 'README.md', commit_message="Update model card")
            print("✓ Model card created and uploaded")

    def save_metrics(self, epoch=None):
        """Save training metrics to JSON."""
        metrics = {
            'epochs': list(range(1, len(self.train_losses) + 1)),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'learning_rates': self.learning_rates,
            'best_test_accuracy': self.best_test_acc
        }

        metrics_path = os.path.join(self.checkpoint_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        if self.hf_api and epoch:
            commit_msg = f"Update metrics (epoch {epoch})" if epoch else "Update metrics"
            self._upload_to_huggingface(metrics_path, 'metrics.json', commit_message=commit_msg)

    def print_model_summary(self):
        """Print the model architecture summary."""
        print("\n" + "="*70)
        print("Model Architecture Summary")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        print(f"MixUp: {self.use_mixup} (alpha={self.mixup_alpha})")
        print(f"Label Smoothing: {self.label_smoothing}")

        # Calculate model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")

        print("\nModel Summary:")

        # Use appropriate input size based on dataset
        # torchinfo uses: (batch, channels, height, width)
        # torchsummary uses: (channels, height, width)
        torchinfo_input_size = (1, 3, 224, 224) if self.dataset == 'imagenet' else (1, 3, 32, 32)
        torchsummary_input_size = (3, 224, 224) if self.dataset == 'imagenet' else (3, 32, 32)

        # Try to use torchinfo.summary if available (preferred)
        if TORCHINFO_AVAILABLE:
            try:
                torchinfo_summary(self.model, input_size=torchinfo_input_size, device=str(self.device))
            except Exception as e:
                print(f"\n⚠ torchinfo.summary failed: {e}")
                # Try torchsummary as fallback
                if TORCHSUMMARY_AVAILABLE:
                    print("Falling back to torchsummary...\n")
                    try:
                        torchsummary_summary(self.model, input_size=torchsummary_input_size)
                    except Exception as e2:
                        print(f"\n⚠ torchsummary.summary also failed: {e2}")
                else:
                    print("⚠ No fallback available (torchsummary not installed)")
        # Use torchsummary if torchinfo is not available
        elif TORCHSUMMARY_AVAILABLE:
            try:
                torchsummary_summary(self.model, input_size=torchsummary_input_size)
            except Exception as e:
                print(f"\n⚠ torchsummary.summary failed: {e}")
        else:
            print("\n⚠ No model summary libraries available.")
            print("   Install with: pip install torchinfo  OR  pip install torchsummary")

        print("="*70 + "\n")

    def plot_metrics(self):
        """Plot training and testing metrics (loss and accuracy)."""
        epochs = list(range(1, len(self.train_losses) + 1))

        # Print console-based plots
        print("\n" + "="*80)
        print("TRAINING AND TESTING LOSS")
        print("="*80)

        # Create loss plot
        fig = plotille.Figure()
        fig.width = 70
        fig.height = 20
        fig.color_mode = 'byte'
        fig.set_x_limits(min_=1, max_=len(epochs))
        fig.set_y_limits(min_=0, max_=max(max(self.train_losses), max(self.test_losses)) * 1.1)

        fig.plot(epochs, self.train_losses, lc=25, label='Training Loss')
        fig.plot(epochs, self.test_losses, lc=196, label='Testing Loss')

        print(fig.show(legend=True))

        print("\n" + "="*80)
        print("TRAINING AND TESTING ACCURACY")
        print("="*80)

        # Create accuracy plot
        fig = plotille.Figure()
        fig.width = 70
        fig.height = 20
        fig.color_mode = 'byte'
        fig.set_x_limits(min_=1, max_=len(epochs))
        fig.set_y_limits(min_=min(min(self.train_accuracies), min(self.test_accuracies)) * 0.95,
                         max_=100)

        fig.plot(epochs, self.train_accuracies, lc=25, label='Training Accuracy')
        fig.plot(epochs, self.test_accuracies, lc=196, label='Testing Accuracy')

        print(fig.show(legend=True))
        print("="*80 + "\n")

        # Save matplotlib plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True)

        axs[1, 0].plot(self.train_accuracies)
        axs[1, 0].set_title("Training Accuracy")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Accuracy (%)")
        axs[1, 0].grid(True)

        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].grid(True)

        axs[1, 1].plot(self.test_accuracies)
        axs[1, 1].set_title("Test Accuracy")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Accuracy (%)")
        axs[1, 1].grid(True)

        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(self.checkpoint_dir, 'training_curves.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved as '{plot_filename}'\n")

        if self.hf_api:
            self._upload_to_huggingface(plot_filename, 'training_curves.png',
                                        commit_message="Upload training curves")

        plt.close()

    def run(self):
        """Run the complete training process for all epochs."""
        if self.dataset == 'imagenet':
            dataset_name = f"ImageNet ({self.num_classes} classes)"
        else:
            dataset_name = "CIFAR-100"
        print(f"\nTraining {self.model_name} for {self.epochs} epochs on {dataset_name}")
        print("="*70)

        # Print model summary before training
        self.print_model_summary()

        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Test
            test_loss, test_acc = self.test()

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            # Save best model
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                print(f"*** New best model! Test Accuracy: {self.best_test_acc:.2f}% ***")
                self.save_checkpoint(
                    epoch, train_acc, test_acc, train_loss, test_loss,
                    'best_model.pth', is_best=True
                )

            # Save checkpoint at breakpoints
            if epoch in self.checkpoint_epochs:
                checkpoint_name = f'checkpoint_epoch{epoch}.pth'
                print(f"📍 Breakpoint checkpoint at epoch {epoch}")
                self.save_checkpoint(
                    epoch, train_acc, test_acc, train_loss, test_loss,
                    checkpoint_name
                )

            # Save metrics periodically
            if epoch % 10 == 0 or epoch in self.checkpoint_epochs:
                self.save_metrics(epoch)

            print(f"Best Test Accuracy so far: {self.best_test_acc:.2f}%\n")

        # Save final model
        print("\n📦 Saving final model...")
        self.save_checkpoint(
            self.epochs, train_acc, test_acc, train_loss, test_loss,
            'final_model.pth'
        )

        # Save final metrics
        self.save_metrics()

        # Create model card
        if self.hf_api:
            self.create_model_card()

        print(f"\nTraining completed. Best test accuracy: {self.best_test_acc:.2f}%")

        # Plot metrics
        self.plot_metrics()

        return self.best_test_acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train image classification models on ImageNet or CIFAR-100')

    # Dataset and model configuration
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'cifar100'],
                        help='Dataset to train on (default: imagenet)')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'wideresnet', 'net'],
                        help='Model architecture (default: resnet50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory (ImageNet root or CIFAR-100 dir) (default: ./data)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    # Learning rate configuration
    parser.add_argument('--initial-lr', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--max-lr', type=float, default=0.1,
                        help='Maximum learning rate after warmup (default: 0.1)')
    parser.add_argument('--min-lr', type=float, default=1e-4,
                        help='Minimum learning rate (default: 1e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs (default: 5)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'onecycle'],
                        help='Learning rate scheduler (default: cosine)')

    # Regularization
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Weight decay (default: 1e-3)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default: 1.0)')

    # Data augmentation
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable MixUp augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='MixUp alpha parameter (default: 0.2)')

    # Training optimizations
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory (default: ./checkpoints)')
    parser.add_argument('--checkpoint-epochs', type=int, nargs='+',
                        default=[10, 25, 50, 75],
                        help='Epochs to save checkpoints (default: 10 25 50 75)')

    # HuggingFace
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace API token (optional)')
    parser.add_argument('--hf-repo', type=str, default=None,
                        help='HuggingFace repository ID (optional)')

    # WideResNet-specific arguments
    parser.add_argument('--depth', type=int, default=28,
                        help='WideResNet depth (default: 28)')
    parser.add_argument('--widen-factor', type=int, default=10,
                        help='WideResNet width factor (default: 10)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='WideResNet dropout rate (default: 0.3)')

    args = parser.parse_args()

    # Prepare model kwargs
    model_kwargs = {}
    if args.model == 'wideresnet':
        model_kwargs = {
            'depth': args.depth,
            'widen_factor': args.widen_factor,
            'dropRate': args.dropout
        }

    # Create trainer
    trainer = ImageClassificationTrainer(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        initial_lr=args.initial_lr,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        scheduler_type=args.scheduler,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip,
        use_mixup=not args.no_mixup,
        mixup_alpha=args.mixup_alpha,
        use_mixed_precision=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_epochs=args.checkpoint_epochs,
        hf_token=args.hf_token,
        hf_repo=args.hf_repo,
        **model_kwargs
    )

    # Run training
    trainer.run()
