# AWS Multi-GPU Training Setup Guide

This guide shows how to run your ResNet-50 ImageNet training on AWS with multiple GPUs using DistributedDataParallel.

## 🚀 Quick Start

### 1. Launch Multi-GPU Instance

**Option A: Use existing spot script (modify for multi-GPU)**
```bash
# Modify infra/create-spot.sh to use multi-GPU instance types
# Examples: p3.8xlarge (4x V100), p3.16xlarge (8x V100), g5.12xlarge (4x A10G)
```

**Option B: Launch manually**
```bash
# Launch p3.8xlarge (4x V100 GPUs) spot instance
aws ec2 run-instances \
  --instance-type 'p3.8xlarge' \
  --key-name 'your-key-name' \
  --security-group-ids 'sg-your-security-group' \
  --subnet-id 'subnet-your-subnet' \
  --iam-instance-profile '{"Name":"your-iam-role"}' \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"2.00","SpotInstanceType":"one-time"}}' \
  --placement '{"AvailabilityZone":"us-east-1a"}' \
  --image-id 'ami-0cc0caf8555831ce4' \
  --tag-specifications '{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"resnet50-multigpu"},{"Key":"project","Value":"imagenet-training"}]}' \
  --count '1' \
  --region us-east-1
```

### 2. Attach Data Volume

```bash
# Create 500GB EBS volume for ImageNet dataset
VOLUME_ID=$(aws ec2 create-volume \
  --size 500 \
  --availability-zone us-east-1a \
  --volume-type gp3 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=imagenet-multigpu-data}]' \
  --query 'VolumeId' --output text)

# Attach to instance
aws ec2 attach-volume \
  --volume-id $VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/xvdf
```

### 3. Setup Instance

SSH into your instance and run:

```bash
# Mount data volume
sudo mkdir -p /mnt/imagenet
sudo mount /dev/nvme2n1 /mnt/imagenet  # Check with lsblk first
sudo chown ubuntu:ubuntu /mnt/imagenet

# Install dependencies
sudo apt update
sudo apt install -y git htop nvidia-smi

# Clone your repository
cd /mnt/imagenet
git clone https://github.com/yourusername/resnet50-imagenet-1k.git
cd resnet50-imagenet-1k

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Authenticate with HuggingFace
huggingface-cli login
```

### 4. Download Dataset

```bash
# Download ImageNet-1k dataset
python infra/dataset/download_imagenet.py
```

## 🎯 Multi-GPU Training Commands

### Single Node Multi-GPU (Recommended)

```bash
# 4-GPU training with SyncBatchNorm
torchrun --nproc_per_node=4 train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir /mnt/imagenet/dataset \
  --epochs 90 \
  --batch-size 512 \
  --multi-gpu \
  --sync-bn \
  --scheduler cosine \
  --augmentation strong \
  --hf-token $HF_TOKEN \
  --hf-repo your-username/imagenet-resnet50

# 8-GPU training (p3.16xlarge)
torchrun --nproc_per_node=8 train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir /mnt/imagenet/dataset \
  --epochs 90 \
  --batch-size 1024 \
  --multi-gpu \
  --sync-bn \
  --scheduler cosine \
  --augmentation strong \
  --hf-token $HF_TOKEN \
  --hf-repo your-username/imagenet-resnet50
```

### With LR Finder

```bash
# Find optimal learning rates first
torchrun --nproc_per_node=4 train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir /mnt/imagenet/dataset \
  --epochs 90 \
  --batch-size 512 \
  --multi-gpu \
  --sync-bn \
  --scheduler onecycle \
  --lr-finder \
  --augmentation strong \
  --hf-token $HF_TOKEN \
  --hf-repo your-username/imagenet-resnet50
```

### Background Training with Logging

```bash
# Run training in background with logging
nohup torchrun --nproc_per_node=4 train.py \
  --model resnet50-pytorch \
  --dataset imagenet-1k \
  --data-dir /mnt/imagenet/dataset \
  --epochs 90 \
  --batch-size 512 \
  --multi-gpu \
  --sync-bn \
  --scheduler cosine \
  --augmentation strong \
  --hf-token $HF_TOKEN \
  --hf-repo your-username/imagenet-resnet50 \
  >> training.log 2>&1 &

# Monitor training
tail -f training.log
```

## 📊 Instance Types & Recommendations

| Instance Type | GPUs | GPU Type | Memory | vCPUs | Price/Hour* | Best For |
|---------------|------|----------|--------|-------|-------------|----------|
| `p3.2xlarge` | 1x | V100 | 16GB | 8 | ~$3.00 | Single GPU testing |
| `p3.8xlarge` | 4x | V100 | 32GB | 32 | ~$12.00 | **Recommended** |
| `p3.16xlarge` | 8x | V100 | 32GB | 64 | ~$24.00 | Large-scale training |
| `g5.12xlarge` | 4x | A10G | 24GB | 48 | ~$4.00 | Cost-effective |
| `g5.24xlarge` | 4x | A10G | 24GB | 96 | ~$8.00 | High memory needs |

*Spot prices vary by region and time

## 🔧 Configuration Tips

### Batch Size Guidelines
- **Single GPU**: 128-256
- **4 GPUs**: 512-1024 (128-256 per GPU)
- **8 GPUs**: 1024-2048 (128-256 per GPU)

### Memory Optimization
```bash
# Reduce batch size if OOM
--batch-size 256

# Disable mixed precision if needed
--no-amp

# Use gradient accumulation for very large models
# (implement in trainer if needed)
```

### Data Loading Optimization
```bash
# Increase data loader workers
# Edit config.json:
{
  "data_loader": {
    "num_workers": 8,
    "num_workers_lr_finder": 4
  }
}
```

## 📈 Monitoring & Debugging

### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi -l 1

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### Training Monitoring
```bash
# Monitor training logs
tail -f training.log

# Check checkpoint directory
ls -la checkpoint_*/

# Monitor system resources
htop
```

### Common Issues & Solutions

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--batch-size 128

# Disable mixed precision
--no-amp
```

**2. DDP Initialization Failed**
```bash
# Check if launched with torchrun
torchrun --nproc_per_node=4 train.py --multi-gpu ...

# Verify GPU visibility
nvidia-smi
```

**3. Slow Data Loading**
```bash
# Increase workers in config.json
"num_workers": 8

# Use faster storage (EBS gp3 with high IOPS)
```

## 💰 Cost Optimization

### Spot Instances
- Use spot instances for 60-90% cost savings
- Set appropriate max price (check current spot prices)
- Use multiple regions for better availability

### Storage Optimization
- Use EBS gp3 volumes (cheaper than gp2)
- Set appropriate IOPS (1000-3000 for ImageNet)
- Use lifecycle policies for old checkpoints

### Training Optimization
- Use mixed precision (`--no-amp` only if needed)
- Optimize batch size for your GPU memory
- Use gradient checkpointing for large models

## 🚀 Production Deployment

### Using AWS Batch (Optional)
```bash
# Create job definition for multi-GPU training
aws batch register-job-definition \
  --job-definition-name resnet50-multigpu \
  --type container \
  --container-properties '{
    "image": "your-ecr-repo:latest",
    "vcpus": 64,
    "memory": 244000,
    "jobRoleArn": "arn:aws:iam::account:role/batch-job-role",
    "resourceRequirements": [
      {"type": "GPU", "value": "4"}
    ]
  }'
```

### Using SageMaker (Optional)
```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='.',
    role='SageMakerRole',
    instance_type='ml.p3.8xlarge',  # 4x V100
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'model': 'resnet50-pytorch',
        'dataset': 'imagenet-1k',
        'epochs': 90,
        'batch-size': 512,
        'multi-gpu': True,
        'sync-bn': True
    }
)
```

## 📋 Checklist

Before starting training:

- [ ] Instance launched with sufficient GPUs
- [ ] EBS volume attached and mounted
- [ ] Dataset downloaded to `/mnt/imagenet/dataset`
- [ ] Python environment setup with dependencies
- [ ] HuggingFace authentication configured
- [ ] Repository cloned and code updated
- [ ] Batch size configured for GPU memory
- [ ] Monitoring tools ready (`nvidia-smi`, `htop`)
- [ ] Logging configured for background training
- [ ] Checkpoint directory permissions set

## 🎯 Expected Performance

**p3.8xlarge (4x V100) Training Times:**
- ImageNet-1k (90 epochs): ~8-12 hours
- ImageNette (20 epochs): ~30 minutes
- Batch size 512: ~2.5 hours per epoch

**Memory Usage:**
- Model: ~1GB per GPU
- Batch size 128: ~8GB GPU memory
- Batch size 256: ~12GB GPU memory

Your multi-GPU setup is now ready for production training on AWS! 🚀
