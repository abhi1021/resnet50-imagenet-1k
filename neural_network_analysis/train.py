"""Modular training runner for tiny-imagenet using neural_network_analysis modules.

This script is intentionally minimal — the heavy lifting lives in the
`neural_network_analysis` package (data.py, model.py, trainer.py, utils.py).
"""
import os
import argparse
import torch.optim as optim

from neural_network_analysis import data, model as model_mod, trainer, utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-dir', default='./tiny-imagenet-200/train')
    p.add_argument('--val-dir', default='./tiny-imagenet-200/val')
    p.add_argument('--model-dir', default='./saved_model')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--img-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--num-classes', type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    train_dir = os.path.abspath(args.train_dir)
    val_dir = os.path.abspath(args.val_dir)
    model_dir = os.path.abspath(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    device = utils.get_device()
    pin_memory = (device.type == 'cuda')

    train_loader, val_loader, train_ds = data.get_dataloaders(
        train_dir, val_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers, pin_memory=pin_memory
    )

    # Visualize a small sample from the training loader using the helper in utils
    try:
        from torchvision import utils as tv_utils
        dataiter = iter(train_loader)
        imgs, labels = next(dataiter)
        grid = tv_utils.make_grid(imgs[:4])
        utils.imshow_batch(grid)
    except Exception as e:
        # Don't crash if visualization backend is not available (e.g., headless)
        print("Batch visualization skipped:", e)

    model = model_mod.build_resnet('resnet18', num_classes=args.num_classes, pretrained=False)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = None

    best_acc = 0.0
    # Metrics history (to match original script)
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_losses_epoch = []

    for epoch in range(args.epochs):
        print(f'EPOCH: {epoch}')
        t_loss, t_acc, batch_losses, batch_accs = trainer.train_epoch(model, device, train_loader, optimizer, scheduler=scheduler)
        v_loss, v_acc, v_batch_losses, v_batch_accs = trainer.evaluate(model, device, val_loader)

        # Append per-batch data to global lists (original code appended per-batch for some lists)
        train_losses.extend(batch_losses)
        train_acc.extend(batch_accs)
        test_losses.extend(v_batch_losses)
        test_acc.extend(v_batch_accs)

        # Epoch level
        train_losses_epoch.append(t_loss)

        print(f'Epoch {epoch}: Train loss {t_loss:.4f}, Train acc {t_acc:.2f}%, Val loss {v_loss:.4f}, Val acc {v_acc:.2f}%')

        best_acc = trainer.save_checkpoint_if_best(model, optimizer, epoch, v_acc, best_acc, model_dir)


if __name__ == '__main__':
    main()