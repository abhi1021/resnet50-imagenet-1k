"""
Checkpoint management for saving and loading model state.
"""
import os
import re
import torch
import numpy as np
from datetime import datetime


class CheckpointManager:
    """Manages model checkpoints with automatic folder creation and versioning."""

    def __init__(self, base_dir='.', checkpoint_prefix='checkpoint', keep_last_n=5):
        """
        Initialize checkpoint manager.

        Args:
            base_dir: Base directory for checkpoints
            checkpoint_prefix: Prefix for checkpoint folders
            keep_last_n: Number of recent epoch checkpoints to keep (default: 5)
                        -1 = keep all, 0 = keep only breakpoints
        """
        self.base_dir = base_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.keep_last_n = keep_last_n
        self.checkpoint_dir = self._get_next_checkpoint_folder()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"📁 Checkpoint folder: {self.checkpoint_dir}")
        if keep_last_n == -1:
            print(f"   Keep all epoch checkpoints")
        elif keep_last_n == 0:
            print(f"   Keep only breakpoint checkpoints")
        else:
            print(f"   Keep last {keep_last_n} epoch checkpoints + breakpoints")

    def _get_next_checkpoint_folder(self):
        """
        Find the next available checkpoint folder in sequence.

        Returns:
            str: Path to the next checkpoint folder (e.g., './checkpoint_3')
        """
        # Get all directories in base path
        existing_dirs = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]

        # Filter for checkpoint folders matching pattern checkpoint_N
        checkpoint_pattern = re.compile(f'^{self.checkpoint_prefix}_(\\d+)$')
        checkpoint_numbers = []

        for dirname in existing_dirs:
            match = checkpoint_pattern.match(dirname)
            if match:
                checkpoint_numbers.append(int(match.group(1)))

        # Find next available number
        if checkpoint_numbers:
            next_num = max(checkpoint_numbers) + 1
        else:
            next_num = 1

        return os.path.join(self.base_dir, f'{self.checkpoint_prefix}_{next_num}')

    def save_checkpoint(self, model, optimizer, epoch, metrics, config, is_best=False, filename=None):
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch number
            metrics: Dict of metrics (train_acc, test_acc, train_loss, test_loss)
            config: Training configuration dict
            is_best: Whether this is the best model so far
            filename: Optional custom filename (default: best_model.pth or checkpoint_epochN.pth)

        Returns:
            str: Path to saved checkpoint
        """
        if filename is None:
            filename = 'best_model.pth' if is_best else f'checkpoint_epoch{epoch}.pth'

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracy': metrics.get('train_acc', 0.0),
            'test_accuracy': metrics.get('test_acc', 0.0),
            'train_loss': metrics.get('train_loss', 0.0),
            'test_loss': metrics.get('test_loss', 0.0),
            'timestamp': datetime.now().isoformat(),
            'config': config
        }

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            print(f"💾 Saved best model: {checkpoint_path}")
        else:
            print(f"💾 Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, filename='best_model.pth'):
        """
        Load checkpoint from file.

        Args:
            filename: Checkpoint filename to load

        Returns:
            dict: Checkpoint dictionary

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        return checkpoint

    def get_checkpoint_dir(self):
        """Get the current checkpoint directory path."""
        return self.checkpoint_dir

    def set_checkpoint_dir(self, checkpoint_dir):
        """
        Override the checkpoint directory path.

        Args:
            checkpoint_dir: Path to use for checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"📁 Using checkpoint folder: {self.checkpoint_dir}")

    def list_checkpoints(self):
        """
        List all checkpoint files in the checkpoint directory.

        Returns:
            list: List of checkpoint filenames
        """
        if not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = [f for f in os.listdir(self.checkpoint_dir)
                      if f.endswith('.pth')]
        return sorted(checkpoints)

    def save_training_state(self, model, optimizer, scheduler, epoch, metrics,
                           config, metrics_tracker=None, scaler=None, filename=None):
        """
        Save complete training state for resumption.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            metrics: Dict of current metrics (train_acc, test_acc, train_loss, test_loss)
            config: Training configuration dict
            metrics_tracker: MetricsTracker instance (optional)
            scaler: GradScaler instance for AMP (optional)
            filename: Optional custom filename (default: training_state_epoch{N}.pth)

        Returns:
            str: Path to saved checkpoint
        """
        if filename is None:
            filename = f'training_state_epoch{epoch}.pth'

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        # Save complete training state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_accuracy': metrics.get('train_acc', 0.0),
            'test_accuracy': metrics.get('test_acc', 0.0),
            'train_loss': metrics.get('train_loss', 0.0),
            'test_loss': metrics.get('test_loss', 0.0),
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
            }
        }

        # Add CUDA RNG state if available
        if torch.cuda.is_available():
            checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state_all()

        # Add GradScaler state for AMP
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Add MetricsTracker state
        if metrics_tracker is not None:
            checkpoint['metrics_tracker'] = {
                'train_losses': metrics_tracker.train_losses,
                'train_accuracies': metrics_tracker.train_accuracies,
                'test_losses': metrics_tracker.test_losses,
                'test_accuracies': metrics_tracker.test_accuracies,
                'learning_rates': metrics_tracker.learning_rates,
                'best_test_acc': metrics_tracker.best_test_acc,
                'best_epoch': metrics_tracker.best_epoch
            }

        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved training state: {checkpoint_path}")

        return checkpoint_path

    def load_training_state(self, checkpoint_path):
        """
        Load complete training state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            dict: Complete checkpoint dictionary

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✓ Loaded training state from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Test Accuracy: {checkpoint['test_accuracy']:.2f}%")
        print(f"  Timestamp: {checkpoint['timestamp']}")

        return checkpoint

    @staticmethod
    def find_latest_epoch_checkpoint(checkpoint_dir):
        """
        Find the checkpoint with the highest epoch number in a directory.

        Args:
            checkpoint_dir: Directory to search for checkpoints

        Returns:
            str: Path to the latest checkpoint, or None if no checkpoints found
        """
        if not os.path.exists(checkpoint_dir):
            return None

        # Pattern to match training_state_epoch{N}.pth
        pattern = re.compile(r'^training_state_epoch(\d+)\.pth$')

        checkpoints = []
        for filename in os.listdir(checkpoint_dir):
            match = pattern.match(filename)
            if match:
                epoch_num = int(match.group(1))
                checkpoints.append((epoch_num, filename))

        if not checkpoints:
            return None

        # Sort by epoch number and get the latest
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest_epoch, latest_filename = checkpoints[0]

        latest_path = os.path.join(checkpoint_dir, latest_filename)
        print(f"🔍 Found latest checkpoint at epoch {latest_epoch}: {latest_path}")

        return latest_path

    def cleanup_old_checkpoints(self, current_epoch, checkpoint_epochs):
        """
        Remove old training state checkpoints, keeping only:
        1. Last N non-breakpoint checkpoints (based on keep_last_n)
        2. All breakpoint epoch checkpoints (preserved forever)

        Args:
            current_epoch: Current epoch number
            checkpoint_epochs: List of breakpoint epochs to preserve
        """
        # If keep_last_n is -1, keep all checkpoints
        if self.keep_last_n == -1:
            return

        # Get all training state checkpoint files
        pattern = re.compile(r'^training_state_epoch(\d+)\.pth$')
        checkpoints = []

        for filename in os.listdir(self.checkpoint_dir):
            match = pattern.match(filename)
            if match:
                epoch_num = int(match.group(1))
                checkpoints.append((epoch_num, filename))

        if not checkpoints:
            return

        # Sort by epoch number (descending)
        checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Separate breakpoint and non-breakpoint checkpoints
        breakpoint_set = set(checkpoint_epochs)
        non_breakpoint_checkpoints = []

        for epoch_num, filename in checkpoints:
            if epoch_num not in breakpoint_set:
                non_breakpoint_checkpoints.append((epoch_num, filename))

        # Keep only last N non-breakpoint checkpoints
        if self.keep_last_n == 0:
            checkpoints_to_delete = non_breakpoint_checkpoints
        else:
            checkpoints_to_delete = non_breakpoint_checkpoints[self.keep_last_n:]

        # Delete old checkpoints
        for epoch_num, filename in checkpoints_to_delete:
            filepath = os.path.join(self.checkpoint_dir, filename)
            try:
                os.remove(filepath)
                print(f"🗑️  Removed old checkpoint: {filename}")
            except Exception as e:
                print(f"⚠️  Could not remove {filename}: {e}")
