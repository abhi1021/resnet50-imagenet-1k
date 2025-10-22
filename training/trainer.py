"""
Main Trainer class for model training with advanced features.
"""
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
from torchvision import datasets

from utils import CheckpointManager, MetricsTracker, plot_training_curves, get_device
from utils import HuggingFaceUploader, create_model_card, plot_lr_finder
from .lr_finder import LRFinder


class Trainer:
    """
    Main trainer class handling the complete training pipeline.
    """

    def __init__(self, model, train_loader, test_loader, optimizer, scheduler,
                 device=None, checkpoint_manager=None, metrics_tracker=None,
                 scheduler_type='onecycle', use_mixup=True, mixup_alpha=0.2,
                 label_smoothing=0.1, use_amp=True, gradient_clip=1.0,
                 hf_uploader=None, model_name='model', config=None):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            test_loader: Test DataLoader
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            device: Device to train on (auto-detect if None)
            checkpoint_manager: CheckpointManager instance
            metrics_tracker: MetricsTracker instance
            scheduler_type: Type of scheduler ('onecycle' or 'cosine')
            use_mixup: Enable MixUp augmentation
            mixup_alpha: MixUp alpha parameter
            label_smoothing: Label smoothing parameter
            use_amp: Use automatic mixed precision
            gradient_clip: Gradient clipping max norm
            hf_uploader: HuggingFaceUploader instance (optional)
            model_name: Model name for logging
            config: Training configuration dict
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.model.to(self.device)

        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.metrics_tracker = metrics_tracker or MetricsTracker()

        self.scheduler_type = scheduler_type.lower()
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.hf_uploader = hf_uploader
        self.model_name = model_name
        self.config = config or {}

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

    def mixup_data(self, x, y, alpha=0.2):
        """Apply MixUp data augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (avg_loss, accuracy)
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
            if self.use_amp:
                with autocast(device_type=self.device.type):
                    if self.use_mixup:
                        inputs, targets_a, targets_b, lam = self.mixup_data(
                            data, target, alpha=self.mixup_alpha
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
                # Standard training without AMP
                if self.use_mixup:
                    inputs, targets_a, targets_b, lam = self.mixup_data(
                        data, target, alpha=self.mixup_alpha
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

            # Update learning rate for OneCycleLR (per batch)
            if self.scheduler_type == 'onecycle':
                self.scheduler.step()

            # Accuracy tracking
            _, pred = outputs.max(1)
            if self.use_mixup:
                correct += lam * pred.eq(targets_a).sum().item() + (1 - lam) * pred.eq(targets_b).sum().item()
            else:
                correct += pred.eq(target).sum().item()
            processed += len(data)
            epoch_loss += loss.item()

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/processed:.2f}%',
                'lr': f'{current_lr:.6f}'
            })

        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100. * correct / processed
        return avg_loss, accuracy

    def test(self):
        """
        Test the model.

        Returns:
            tuple: (test_loss, accuracy)
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

    def resume_from_checkpoint(self, checkpoint_path):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            int: Epoch to start training from (checkpoint epoch + 1)
        """
        checkpoint = self.checkpoint_manager.load_training_state(checkpoint_path)

        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Restored model state")

        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Restored optimizer state")

        # Restore scheduler state
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Restored scheduler state")

            # Verify scheduler state and show current position
            scheduler_last_epoch = self.scheduler.last_epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Scheduler last_epoch: {scheduler_last_epoch} ({self.scheduler_type.upper()} step counter)")
            print(f"  Current learning rate: {current_lr:.6f}")

        # Restore GradScaler state for AMP
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✓ Restored GradScaler state")

        # Restore MetricsTracker state
        if checkpoint.get('metrics_tracker'):
            mt = checkpoint['metrics_tracker']
            self.metrics_tracker.train_losses = mt['train_losses']
            self.metrics_tracker.train_accuracies = mt['train_accuracies']
            self.metrics_tracker.test_losses = mt['test_losses']
            self.metrics_tracker.test_accuracies = mt['test_accuracies']
            self.metrics_tracker.learning_rates = mt['learning_rates']
            self.metrics_tracker.best_test_acc = mt['best_test_acc']
            self.metrics_tracker.best_epoch = mt['best_epoch']
            print(f"✓ Restored metrics history ({len(mt['train_losses'])} epochs)")

        # Restore RNG states for reproducibility
        if checkpoint.get('rng_state'):
            rng = checkpoint['rng_state']
            torch.set_rng_state(rng['torch'])
            np.random.set_state(rng['numpy'])
            if 'cuda' in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng['cuda'])
            print("✓ Restored RNG states")

        start_epoch = checkpoint['epoch'] + 1

        # Display detailed last epoch state information
        print(f"\n{'='*70}")
        print(f"LAST TRAINING STATE (Epoch {checkpoint['epoch']})")
        print(f"{'='*70}")
        print(f"Training:")
        print(f"  Loss: {checkpoint.get('train_loss', 0.0):.4f}")
        print(f"  Accuracy: {checkpoint.get('train_accuracy', 0.0):.2f}%")
        print(f"Test:")
        print(f"  Loss: {checkpoint.get('test_loss', 0.0):.4f}")
        print(f"  Accuracy: {checkpoint.get('test_accuracy', 0.0):.2f}%")

        # Get current LR from optimizer (already restored)
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"{'='*70}\n")

        print(f"{'='*70}")
        print(f"RESUMING TRAINING FROM EPOCH {start_epoch}")
        print(f"{'='*70}")
        print(f"Previous best accuracy: {self.metrics_tracker.best_test_acc:.2f}% (epoch {self.metrics_tracker.best_epoch})")
        print(f"{'='*70}\n")

        return start_epoch

    def run(self, epochs, patience=15, checkpoint_epochs=None, target_accuracy=None, start_epoch=1):
        """
        Run the complete training loop.

        Args:
            epochs: Total number of epochs to train
            patience: Early stopping patience
            checkpoint_epochs: List of epochs to save as breakpoint checkpoints (kept forever)
            target_accuracy: Target accuracy to stop training
            start_epoch: Epoch to start from (default: 1, >1 when resuming)

        Returns:
            float: Best test accuracy achieved
        """
        checkpoint_epochs = checkpoint_epochs or [10, 20, 25, 30, 40, 50, 60, 75, 90]
        patience_counter = 0

        # Save config.json at the start of training (only for epoch 1)
        if start_epoch == 1:
            import json
            checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir()
            config_path = f"{checkpoint_dir}/config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✓ Saved initial config: {config_path}\n")

        for epoch in range(start_epoch, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            test_loss, test_acc = self.test()

            # Step scheduler after epoch for CosineAnnealing (OneCycle steps per batch)
            if self.scheduler_type == 'cosine':
                self.scheduler.step()

            # Check if best model BEFORE updating tracker
            current_lr = self.optimizer.param_groups[0]['lr']
            is_best = test_acc > self.metrics_tracker.best_test_acc

            # Track metrics (this will update best_test_acc)
            self.metrics_tracker.update(train_loss, train_acc, test_loss, test_acc, current_lr)

            # Save best model
            if is_best:
                patience_counter = 0
                print(f"*** New best model! Test Accuracy: {test_acc:.2f}% ***")

                metrics = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, metrics, self.config, is_best=True
                )
            else:
                patience_counter += 1

            # Save training state at EVERY epoch for resumption
            metrics = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_loss': train_loss,
                'test_loss': test_loss
            }
            self.checkpoint_manager.save_training_state(
                self.model, self.optimizer, self.scheduler, epoch, metrics,
                self.config, self.metrics_tracker, self.scaler
            )

            # Save metrics.json after EVERY epoch to ensure it's always up-to-date
            checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir()
            metrics_path = f"{checkpoint_dir}/metrics.json"
            self.metrics_tracker.save(metrics_path)

            # Save training curves after EVERY epoch for visual progress tracking
            from utils import plot_training_curves
            curves_path = f"{checkpoint_dir}/training_curves.png"
            plot_training_curves(self.metrics_tracker, curves_path)

            # Cleanup old checkpoints (keep rolling window + breakpoints)
            self.checkpoint_manager.cleanup_old_checkpoints(epoch, checkpoint_epochs)

            # Mark breakpoint epochs
            if epoch in checkpoint_epochs:
                print(f"📍 Breakpoint checkpoint at epoch {epoch} (will be kept permanently)")

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}. No improvement for {patience} epochs.")
                break

            # Check if target reached
            if target_accuracy and test_acc >= target_accuracy:
                print(f"\n{'=' * 70}")
                print(f"Target accuracy of {target_accuracy}% reached at epoch {epoch}!")
                print(f"Final test accuracy: {test_acc:.2f}%")
                print(f"{'=' * 70}")
                break

            print(f"Best so far: {self.metrics_tracker.best_test_acc:.2f}% | Patience: {patience_counter}/{patience}\n")

        print(f"\nTraining completed. Best test accuracy: {self.metrics_tracker.best_test_acc:.2f}% "
              f"(epoch {self.metrics_tracker.best_epoch})")

        return self.metrics_tracker.best_test_acc

    def save_results(self):
        """Save all training results (metrics, plots, model card)."""
        checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir()

        # Save metrics
        metrics_path = f"{checkpoint_dir}/metrics.json"
        self.metrics_tracker.save(metrics_path)

        # Plot and save training curves
        self.metrics_tracker.print_console_plots()
        curves_path = f"{checkpoint_dir}/training_curves.png"
        plot_training_curves(self.metrics_tracker, curves_path)

        # Save config
        import json
        config_path = f"{checkpoint_dir}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✓ Saved config: {config_path}")

        # Create model card
        if self.hf_uploader:
            model_card = create_model_card(
                self.hf_uploader.repo_id,
                self.model_name,
                self.metrics_tracker.best_test_acc,
                len(self.metrics_tracker.train_losses),
                self.metrics_tracker,
                self.config
            )
            readme_path = f"{checkpoint_dir}/README.md"
            with open(readme_path, 'w') as f:
                f.write(model_card)
            print(f"✓ Saved model card: {readme_path}")

    def upload_to_huggingface(self):
        """Upload results to HuggingFace Hub if configured."""
        if not self.hf_uploader:
            print("\n⚠ HuggingFace upload skipped (no uploader configured)")
            return

        checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir()
        self.hf_uploader.upload_checkpoint_files(checkpoint_dir)

    def run_lr_finder(self, config):
        """
        Run LR Finder to find optimal learning rates.

        Args:
            config: LR Finder configuration dict

        Returns:
            dict: Suggested LR values
        """
        print("\n" + "="*70)
        print("STARTING LR FINDER")
        print("="*70)

        # Create clean data loader (no mixup, no label smoothing)
        from data_loaders.transforms import TestTransformWrapper, ImageNetTestTransform
        from data_loaders import get_dataset_info
        import os

        # Get dataset type and data directory from config
        dataset_name = self.config.get('dataset', 'cifar100')
        data_dir = self.config.get('data_dir', '../data')

        dataset_info = get_dataset_info(dataset_name)

        # Use appropriate transform based on dataset type
        if dataset_name in ['imagenet', 'imagenette', 'imagenet-1k']:
            test_transform = ImageNetTestTransform(dataset_info['mean'], dataset_info['std'])
        else:
            test_transform = TestTransformWrapper(dataset_info['mean'], dataset_info['std'])

        # Create dataset based on type
        if dataset_name == 'imagenet-1k':
            # For HuggingFace ImageNet-1K, use the custom dataset class
            from data_loaders.imagenet_hf import ImageNet1KDataset
            hf_token = self.config.get('hf_token', None)
            clean_dataset = ImageNet1KDataset(
                data_dir=data_dir,
                train=True,
                hf_token=hf_token
            )
        elif dataset_name in ['imagenet', 'imagenette']:
            # For ImageNet-style datasets, use ImageFolder
            train_dir = os.path.join(data_dir, 'train')
            clean_dataset = datasets.ImageFolder(
                train_dir,
                transform=test_transform
            )
        elif dataset_name == 'cifar100':
            clean_dataset = datasets.CIFAR100(
                data_dir,
                train=True,
                download=False,
                transform=test_transform
            )
        elif dataset_name == 'cifar10':
            clean_dataset = datasets.CIFAR10(
                data_dir,
                train=True,
                download=False,
                transform=test_transform
            )
        else:
            raise ValueError(f"Unsupported dataset for LR Finder: {dataset_name}")

        # Only use pin_memory on CUDA devices
        use_pin_memory = self.device.type == 'cuda'

        clean_loader = torch.utils.data.DataLoader(
            clean_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        # Create LR Finder
        criterion = torch.nn.CrossEntropyLoss()
        lr_finder = LRFinder(
            model=self.model,
            optimizer=self.optimizer,
            criterion=criterion,
            device=self.device,
            data_loader=clean_loader
        )

        # Run range test
        num_epochs = config.get('num_epochs', 3)
        start_lr = config.get('start_lr', 1e-6)
        end_lr = config.get('end_lr', 1.0)
        selection_method = config.get('selection_method', 'steepest_gradient')

        lr_finder.range_test(
            start_lr=start_lr,
            end_lr=end_lr,
            num_epochs=num_epochs
        )

        # Get scheduler-specific LR suggestions
        suggested_lrs = lr_finder.suggest_scheduler_lrs(
            scheduler_type=self.scheduler_type,
            method=selection_method
        )

        # Save plot
        checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir()
        plot_path = f"{checkpoint_dir}/lr_finder_plot.png"

        if self.scheduler_type == 'onecycle':
            plot_lr_finder(
                lr_finder.lrs,
                lr_finder.losses,
                plot_path,
                max_lr=suggested_lrs.get('max_lr'),
                base_lr=suggested_lrs.get('base_lr')
            )
        else:
            plot_lr_finder(
                lr_finder.lrs,
                lr_finder.losses,
                plot_path,
                suggested_lr=suggested_lrs.get('initial_lr')
            )

        print(f"\n{'='*70}")
        print(f"LR FINDER COMPLETED")
        print(f"{'='*70}\n")

        return suggested_lrs
