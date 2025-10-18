import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
from huggingface_hub import HfApi
try:
    from torch.distributions import Beta as BetaDist
except Exception:
    BetaDist = None


def train_epoch(model, device, train_loader, optimizer, scheduler=None, scheduler_step_per_batch=False, cutmix_prob=0.0, cutmix_alpha=1.0, label_smoothing=0.0):
    """Run one training epoch.

    Returns:
        epoch_loss (float), epoch_acc (float), batch_losses (list), batch_accs (list)
    """
    model.train()
    correct = 0
    processed = 0
    batch_losses = []
    batch_accs = []

    pbar = tqdm(train_loader, desc='Train', leave=True)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if np.random.rand() < cutmix_prob:
            # CutMix
            # sample lambda using torch Beta if available, otherwise numpy
            if BetaDist is not None:
                lam = BetaDist(cutmix_alpha, cutmix_alpha).sample().item()
            else:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
            batch_size = data.size(0)
            index = torch.randperm(batch_size).to(data.device)
            # bounding box
            H, W = data.size(2), data.size(3)
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            x1 = np.clip(cx - cut_w // 2, 0, W)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            y2 = np.clip(cy + cut_h // 2, 0, H)
            data[:, :, y1:y2, x1:x2] = data[index, :, y1:y2, x1:x2]
            # correct lambda to actual area
            lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
            # Predict
            outputs = model(data)
            # Mix loss
            if label_smoothing and label_smoothing > 0.0:
                loss = lam * F.cross_entropy(outputs, target, label_smoothing=label_smoothing) + (1 - lam) * F.cross_entropy(outputs, target[index], label_smoothing=label_smoothing)
            else:
                loss = lam * F.cross_entropy(outputs, target) + (1 - lam) * F.cross_entropy(outputs, target[index])
        else:
            # Predict
            outputs = model(data)
            # Calculate loss (CrossEntropyLoss accepts raw logits)
            if label_smoothing and label_smoothing > 0.0:
                loss = F.cross_entropy(outputs, target, label_smoothing=label_smoothing)
            else:
                loss = F.cross_entropy(outputs, target)
        
        loss.backward()
        optimizer.step()
        # per-batch scheduler stepping is handled via scheduler_step_per_batch flag
        if scheduler is not None and scheduler_step_per_batch:
            try:
                scheduler.step()
            except TypeError:
                pass

        pred = outputs.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        processed += data.size(0)
        batch_losses.append(loss.item())

        running_acc = 100. * correct / processed if processed > 0 else 0.0
        batch_accs.append(running_acc)

        # Mirror previous behavior: show loss, batch id, and running accuracy
        pbar.set_description(desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={running_acc:0.2f}')

    # batch_losses are per-batch mean losses
    epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
    epoch_acc = 100. * correct / processed if processed > 0 else 0.0
    return epoch_loss, epoch_acc, batch_losses, batch_accs


def evaluate(model, device, loader):
    """Evaluate model on `loader`. Returns (test_loss, acc, batch_losses, batch_accs)."""
    model.eval()
    test_loss = 0
    correct = 0
    batch_losses = []
    batch_accs = []
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc='Eval', leave=True)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum').item()
            test_loss += loss
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            batch_size = data.size(0)
            total += batch_size
            batch_losses.append(loss / batch_size)
            batch_accs.append(100. * pred.eq(target).sum().item() / batch_size)
            pbar.set_description(desc=f'Eval Batch_id={batch_idx} Acc={100.*correct/total:0.2f}')

    test_loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    return test_loss, acc, batch_losses, batch_accs


def save_checkpoint_if_best(model, optimizer, epoch, test_acc, best_acc, model_dir, hf_username=None, model_name=None):
    os.makedirs(model_dir, exist_ok=True)
    if test_acc > best_acc:
        best_acc = test_acc
        filename = os.path.join(model_dir, 'pytorch_model.pt')
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'test_acc': test_acc}, filename)

        # Optionally upload to Hugging Face hub if credentials/config provided
        if hf_username and model_name and best_acc > 50.0:
            repo_id = f"{hf_username}/{model_name}"
            api = HfApi()
            try:
                api.upload_file(path_or_fileobj=filename, path_in_repo=os.path.basename(filename), repo_id=repo_id, repo_type='model')
            except Exception:
                pass

    return best_acc
