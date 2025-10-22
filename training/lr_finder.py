"""
Learning Rate Finder for automatic LR selection.

Based on Leslie Smith's paper: "Cyclical Learning Rates for Training Neural Networks"
"""
import torch
import numpy as np
import json
import os
from tqdm import tqdm


class LRFinder:
    """
    Learning Rate Finder using the range test method.
    Helps find optimal learning rates before training.
    """

    def __init__(self, model, optimizer, criterion, device, data_loader, checkpoint_dir=None):
        """
        Initialize LR Finder.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            criterion: Loss criterion
            device: Device to run on (cuda/mps/cpu)
            data_loader: DataLoader for LR range test
            checkpoint_dir: Directory to save/load progress checkpoints (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.data_loader = data_loader
        self.checkpoint_dir = checkpoint_dir

        # Storage for results
        self.lrs = []
        self.losses = []

        # State management
        self.best_loss = float('inf')
        self.initial_model_state = None
        self.initial_optimizer_state = None

    def _get_progress_path(self):
        """Get the path for the progress checkpoint file."""
        if self.checkpoint_dir is None:
            return None
        return os.path.join(self.checkpoint_dir, 'lr_finder_progress.json')

    def save_progress(self, epoch, iteration, start_lr, end_lr, num_epochs, smoothing):
        """
        Save LR finder progress after each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            iteration: Current iteration number
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_epochs: Total number of epochs
            smoothing: Smoothing factor
        """
        progress_path = self._get_progress_path()
        if progress_path is None:
            return

        progress = {
            'epoch': epoch,
            'iteration': iteration,
            'lrs': self.lrs,
            'losses': self.losses,
            'best_loss': self.best_loss,
            'start_lr': start_lr,
            'end_lr': end_lr,
            'num_epochs': num_epochs,
            'smoothing': smoothing,
            'model_state_dict': {k: v.cpu().tolist() if v.numel() < 1000 else None
                                for k, v in self.model.state_dict().items()},
            'optimizer_state_dict': str(self.optimizer.state_dict())  # Simplified for JSON
        }

        try:
            with open(progress_path, 'w') as f:
                json.dump(progress, f, indent=2)
            print(f"💾 Saved LR finder progress: epoch {epoch+1}/{num_epochs}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save LR finder progress: {e}")

    def load_progress(self):
        """
        Load LR finder progress from previous interrupted run.

        Returns:
            dict: Progress state if found, None otherwise
        """
        progress_path = self._get_progress_path()
        if progress_path is None or not os.path.exists(progress_path):
            return None

        try:
            with open(progress_path, 'r') as f:
                progress = json.load(f)

            print(f"\n{'='*70}")
            print(f"✓ Found interrupted LR finder progress")
            print(f"{'='*70}")
            print(f"  Resuming from epoch {progress['epoch']+1}/{progress['num_epochs']}")
            print(f"  Collected {len(progress['lrs'])} data points so far")
            print(f"{'='*70}\n")

            # Restore state
            self.lrs = progress['lrs']
            self.losses = progress['losses']
            self.best_loss = progress['best_loss']

            return progress
        except Exception as e:
            print(f"⚠️  Warning: Could not load LR finder progress: {e}")
            print(f"   Starting LR finder from scratch\n")
            return None

    def clear_progress(self):
        """Delete progress checkpoint file when LR finder completes successfully."""
        progress_path = self._get_progress_path()
        if progress_path and os.path.exists(progress_path):
            try:
                os.remove(progress_path)
                print(f"✓ Cleared LR finder progress checkpoint")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete progress file: {e}")

    def range_test(self, start_lr=1e-6, end_lr=1.0, num_epochs=3, smoothing=0.05):
        """
        Run the LR range test.

        Args:
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_epochs: Number of epochs to run
            smoothing: Exponential smoothing factor for loss

        Returns:
            tuple: (learning_rates, losses)
        """
        # Check for interrupted progress
        progress = self.load_progress()
        start_epoch = 0
        iteration = 0
        smoothed_loss = 0.0

        if progress:
            # Resume from interrupted run
            start_epoch = progress['epoch'] + 1
            iteration = progress['iteration']
            start_lr = progress['start_lr']
            end_lr = progress['end_lr']
            num_epochs = progress['num_epochs']
            smoothing = progress['smoothing']
            # lrs, losses, best_loss already restored in load_progress()
        else:
            # Fresh start
            print("\n" + "="*70)
            print("LEARNING RATE FINDER - RANGE TEST")
            print("="*70)
            print(f"Testing learning rates from {start_lr:.2e} to {end_lr:.2e}")
            print(f"Running for {num_epochs} epochs")
            print(f"Note: Using clean data (no MixUp, no Label Smoothing)")
            print("="*70 + "\n")

        # Save initial state (only if fresh start)
        if not progress:
            self.initial_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.initial_optimizer_state = self.optimizer.state_dict()

        # Calculate total iterations
        total_iters = len(self.data_loader) * num_epochs
        lr_lambda = lambda x: np.exp(x * np.log(end_lr / start_lr) / total_iters)

        # Set initial LR (or current LR if resuming)
        if not progress:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = start_lr

        # Run training
        self.model.train()

        for epoch in range(start_epoch, num_epochs):
            pbar = tqdm(self.data_loader, desc=f"LR Finder Epoch {epoch+1}/{num_epochs}")

            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass (no mixup, no label smoothing)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target)

                # Track smoothed loss
                if iteration == 0:
                    smoothed_loss = loss.item()
                else:
                    smoothed_loss = smoothing * loss.item() + (1 - smoothing) * smoothed_loss

                # Check for divergence
                if smoothed_loss > 4 * self.best_loss or torch.isnan(loss):
                    print(f"\n⚠ Loss diverging (current: {smoothed_loss:.4f}, best: {self.best_loss:.4f})")
                    print("Stopping LR range test early")
                    break

                # Update best loss
                if smoothed_loss < self.best_loss:
                    self.best_loss = smoothed_loss

                # Store LR and loss
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lrs.append(current_lr)
                self.losses.append(smoothed_loss)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Update learning rate
                iteration += 1
                new_lr = start_lr * lr_lambda(iteration)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

                pbar.set_postfix({'lr': f'{current_lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})

            # Early stopping if loss diverged
            if smoothed_loss > 4 * self.best_loss or torch.isnan(loss):
                break

            # Save progress after each epoch
            self.save_progress(epoch, iteration, start_lr, end_lr, num_epochs, smoothing)

        # Restore initial state
        if self.initial_model_state:
            print("\nRestoring initial model and optimizer state...")
            self.model.load_state_dict(self.initial_model_state)
            self.optimizer.load_state_dict(self.initial_optimizer_state)

        print(f"✓ LR range test completed: {len(self.lrs)} data points collected\n")

        # Clear progress checkpoint on successful completion
        self.clear_progress()

        return self.lrs, self.losses

    def find_steepest_gradient(self, skip_start=10, skip_end=5):
        """
        Find LR with steepest negative gradient (fastest learning).

        Args:
            skip_start: Skip first N points (noisy at very low LR)
            skip_end: Skip last N points (diverging region)

        Returns:
            float: Suggested learning rate
        """
        if len(self.losses) < skip_start + skip_end + 10:
            print("⚠ Not enough data points for gradient calculation")
            return None

        # Calculate gradients
        losses = np.array(self.losses[skip_start:-skip_end if skip_end > 0 else None])
        lrs = np.array(self.lrs[skip_start:-skip_end if skip_end > 0 else None])

        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)

        suggested_lr = lrs[min_gradient_idx]
        print(f"  Method: Steepest Gradient")
        print(f"  → Suggested LR: {suggested_lr:.6f}")
        print(f"  → Loss at this point: {losses[min_gradient_idx]:.4f}")

        return suggested_lr

    def find_before_divergence(self, threshold=0.1, skip_start=10):
        """
        Find LR just before loss starts increasing significantly.

        Args:
            threshold: Percentage increase to consider as divergence
            skip_start: Skip first N points

        Returns:
            float: Suggested learning rate
        """
        if len(self.losses) < skip_start + 10:
            print("⚠ Not enough data points")
            return None

        losses = np.array(self.losses[skip_start:])
        lrs = np.array(self.lrs[skip_start:])

        # Find minimum loss
        min_loss_idx = np.argmin(losses)
        min_loss = losses[min_loss_idx]

        # Find where loss increases by threshold percentage
        for i in range(min_loss_idx, len(losses)):
            if losses[i] > min_loss * (1 + threshold):
                # Go back a bit for safety
                safe_idx = max(0, i - 5)
                suggested_lr = lrs[safe_idx]
                print(f"  Method: Before Divergence")
                print(f"  → Suggested LR: {suggested_lr:.6f}")
                print(f"  → Loss at this point: {losses[safe_idx]:.4f}")
                return suggested_lr

        # If no divergence found, use LR at minimum loss
        suggested_lr = lrs[min_loss_idx]
        print(f"  Method: Before Divergence (no divergence detected)")
        print(f"  → Suggested LR: {suggested_lr:.6f}")
        print(f"  → Loss at this point: {min_loss:.4f}")

        return suggested_lr

    def find_valley(self, skip_start=10, skip_end=5):
        """
        Find LR at the valley (minimum loss point).

        Args:
            skip_start: Skip first N points
            skip_end: Skip last N points

        Returns:
            float: Suggested learning rate
        """
        if len(self.losses) < skip_start + skip_end + 10:
            print("⚠ Not enough data points")
            return None

        losses = np.array(self.losses[skip_start:-skip_end if skip_end > 0 else None])
        lrs = np.array(self.lrs[skip_start:-skip_end if skip_end > 0 else None])

        min_idx = np.argmin(losses)
        suggested_lr = lrs[min_idx]

        print(f"  Method: Valley (Minimum Loss)")
        print(f"  → Suggested LR: {suggested_lr:.6f}")
        print(f"  → Loss at this point: {losses[min_idx]:.4f}")

        return suggested_lr

    def suggest_lr(self, method='steepest_gradient'):
        """
        Suggest learning rate based on the selected method.

        Args:
            method: One of 'steepest_gradient', 'before_divergence', 'valley', 'manual'

        Returns:
            float or None: Suggested learning rate (None for manual)
        """
        print("\n" + "="*70)
        print("LR FINDER ANALYSIS")
        print("="*70)

        if method == 'steepest_gradient':
            return self.find_steepest_gradient()
        elif method == 'before_divergence':
            return self.find_before_divergence()
        elif method == 'valley':
            return self.find_valley()
        elif method == 'manual':
            print(f"  Method: Manual Selection")
            print(f"  → Please inspect the plot and select LR manually")
            print(f"  → LR range tested: {self.lrs[0]:.2e} to {self.lrs[-1]:.2e}")
            return None
        else:
            print(f"⚠ Unknown method: {method}, defaulting to steepest_gradient")
            return self.find_steepest_gradient()

    def suggest_scheduler_lrs(self, scheduler_type, method='steepest_gradient'):
        """
        Suggest learning rates specific to the scheduler type.

        For OneCycle: Suggests both max_lr and base_lr
        For CosineAnnealing: Suggests initial_lr

        Args:
            scheduler_type: 'onecycle' or 'cosine'
            method: LR selection method

        Returns:
            dict: Dictionary with suggested LR values
        """
        if scheduler_type == 'onecycle':
            return self._suggest_onecycle_lrs(method)
        else:  # cosine
            return self._suggest_cosine_lr(method)

    def _suggest_onecycle_lrs(self, method):
        """
        Suggest max_lr and base_lr for OneCycleLR scheduler.

        Strategy:
        1. Find max_lr using the selected method
        2. Find base_lr at an earlier point in the curve (1/10 to 1/5 of max_lr)
        """
        # Find max_lr using the selected method
        max_lr = self.suggest_lr(method)

        if max_lr is None:
            print("⚠ Could not determine max_lr")
            return {'max_lr': None, 'base_lr': None}

        # Find base_lr: look for a safe starting point
        base_lr = max_lr / 10.0  # Default to 1/10

        # Try to find a more optimal base_lr by analyzing early portion of curve
        if len(self.lrs) > 20:
            # Find index of max_lr
            max_lr_idx = min(range(len(self.lrs)), key=lambda i: abs(self.lrs[i] - max_lr))

            # Look at first 30% of the range before max_lr
            early_portion = int(max_lr_idx * 0.3)
            if early_portion > 10:
                # Find where loss starts decreasing significantly
                early_losses = np.array(self.losses[:early_portion])
                early_lrs = np.array(self.lrs[:early_portion])

                # Find steepest descent in early portion
                if len(early_losses) > 5:
                    gradients = np.gradient(early_losses)
                    # Find where significant learning starts (gradient becomes more negative)
                    threshold = np.percentile(gradients, 20)  # 20th percentile of gradients
                    significant_learning_idx = np.where(gradients < threshold)[0]

                    if len(significant_learning_idx) > 0:
                        base_lr_idx = significant_learning_idx[0]
                        base_lr = early_lrs[base_lr_idx]

        print(f"\n{'='*70}")
        print(f"ONECYCLE LR RECOMMENDATIONS")
        print(f"{'='*70}")
        print(f"  Max LR:  {max_lr:.6f} (peak learning rate)")
        print(f"  Base LR: {base_lr:.6f} (starting learning rate)")
        print(f"  Ratio:   {max_lr/base_lr:.1f}x (max_lr / base_lr)")
        print(f"{'='*70}\n")

        return {'max_lr': max_lr, 'base_lr': base_lr}

    def _suggest_cosine_lr(self, method):
        """
        Suggest initial_lr for CosineAnnealingWarmRestarts scheduler.
        """
        initial_lr = self.suggest_lr(method)

        if initial_lr is None:
            print("⚠ Could not determine initial_lr")
            return {'initial_lr': None}

        print(f"\n{'='*70}")
        print(f"COSINE ANNEALING LR RECOMMENDATIONS")
        print(f"{'='*70}")
        print(f"  Initial LR: {initial_lr:.6f}")
        print(f"  This will be used with existing T_0 and eta_min parameters")
        print(f"{'='*70}\n")

        return {'initial_lr': initial_lr}
