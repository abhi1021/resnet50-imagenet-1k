import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
from huggingface_hub import HfApi


def train_epoch(model, device, train_loader, optimizer, scheduler=None, cutmix_prob=0.0):
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
        outputs = model(data)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        pred = outputs.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        processed += data.size(0)
        batch_losses.append(loss.item())

        running_acc = 100. * correct / processed if processed > 0 else 0.0
        batch_accs.append(running_acc)

        # Mirror previous behavior: show loss, batch id, and running accuracy
        pbar.set_description(desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={running_acc:0.2f}')

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
