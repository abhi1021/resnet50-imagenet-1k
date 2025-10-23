# Infrastructure Guide (infra)

This folder contains scripts for dataset access and runtime setup for training ResNet-50 on ImageNet-1k.

## Folder layout
- `dataset/download_imagenet.py` — downloads the ImageNet-1k dataset via Hugging Face Datasets.
- `dataset/mount-volume.sh` — reference commands to mount a data volume, set up an environment, launch training, and monitor.

## Prerequisites
- Python 3.9+ with `pip`
- GPU machine (for training examples) with NVIDIA drivers and CUDA toolkit
- Hugging Face account and access to `ILSVRC/imagenet-1k`
- Optional: a dedicated data disk to mount at `/mnt/imagenet`

## Authentication (Hugging Face)
Authenticate once on the machine where you will download/use the dataset:

```bash
huggingface-cli login
```

Alternatively, export a token (avoid hardcoding secrets):

```bash
export HF_TOKEN={{HF_TOKEN}}
```

## Download ImageNet-1k
This uses the Hugging Face Datasets library to fetch `ILSVRC/imagenet-1k` into the local cache.

```bash
# Ensure dependencies are installed
python -m pip install --upgrade pip
pip install datasets

# Run the downloader
python infra/dataset/download_imagenet.py
```

Notes:
- Default cache: `~/.cache/huggingface/datasets` (override via `HF_DATASETS_CACHE`).
- Make sure your account has access to the dataset.

## Mount a data volume (Linux)
Use these commands as a reference (see `dataset/mount-volume.sh`). Adjust the device path.

```bash
# Inspect block devices and identify the correct one (e.g., /dev/nvme2n1)
lsblk

# Create mount point and mount the volume
sudo mkdir -p /mnt/imagenet
sudo mount /dev/<your_device> /mnt/imagenet

# Set ownership to your user
sudo chown $(id -u):$(id -g) /mnt/imagenet
```

## Python environment (optional but recommended)
```bash
python -m venv /mnt/imagenet/venv
source /mnt/imagenet/venv/bin/activate
pip install -r requirements.txt  # if present in repo
```

## Start training (examples)
From your project root (this repo) or where `train.py` lives. Replace placeholders as needed.

```bash
# Example 1: standard run with logging
nohup python train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir /mnt/imagenet/dataset \
  --epochs 90 \
  --batch-size 170 \
  --scheduler onecycle \
  --lr-finder \
  --resume-from ./checkpoint_1 \
  --hf-token ${HF_TOKEN:-} \
  --hf-repo <your-username/imagenet1k> \
  >> console.log 2>&1 &

# Example 2: alternate hyperparameters
nohup python train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir /mnt/imagenet/dataset \
  --epochs 100 \
  --batch-size 256 \
  --scheduler onecycle \
  --lr-finder \
  --resume-from ./checkpoint_3 \
  --hf-token ${HF_TOKEN:-} \
  --hf-repo <your-username/imagenet1k> \
  >> console-3.log 2>&1 &
```

If you used `huggingface-cli login`, you can omit `--hf-token`.

## Monitoring
```bash
# GPU utilization (Linux)
watch -n 0.5 nvidia-smi

# CPU/memory
htop

# Training logs
tail -f console.log   # or console-3.log
```

## Security notes
- Do NOT hardcode tokens in scripts or logs. Use `huggingface-cli login` or an environment variable like `HF_TOKEN`.
- Redirect logs to files as shown and avoid printing secrets.
