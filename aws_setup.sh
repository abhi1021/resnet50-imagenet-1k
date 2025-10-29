#!/bin/bash
# AWS Multi-GPU Training Setup Script
# Run this on your EC2 instance after mounting the data volume

set -e

echo "🚀 Setting up AWS Multi-GPU Training Environment..."

# Check if running on EC2
if ! curl -s http://169.254.169.254/latest/meta-data/instance-id > /dev/null; then
    echo "❌ This script must be run on an EC2 instance"
    exit 1
fi

# Get instance info
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
echo "📋 Instance: $INSTANCE_TYPE ($INSTANCE_ID)"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "   Found $GPU_COUNT GPU(s)"
else
    echo "⚠️  nvidia-smi not found - ensure NVIDIA drivers are installed"
fi

# Setup directories
echo "📁 Setting up directories..."
sudo mkdir -p /mnt/imagenet
sudo chown ubuntu:ubuntu /mnt/imagenet

# Check if volume is mounted
if ! mountpoint -q /mnt/imagenet; then
    echo "⚠️  /mnt/imagenet is not mounted. Please mount your EBS volume first:"
    echo "   sudo mount /dev/nvme2n1 /mnt/imagenet"
    echo "   sudo chown ubuntu:ubuntu /mnt/imagenet"
    exit 1
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y git htop tree jq

# Setup Python environment
echo "🐍 Setting up Python environment..."
cd /mnt/imagenet

# Clone repository if not exists
if [ ! -d "resnet50-imagenet-1k" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/yourusername/resnet50-imagenet-1k.git
fi

cd resnet50-imagenet-1k

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check HuggingFace authentication
echo "🔐 Checking HuggingFace authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "⚠️  HuggingFace not authenticated. Please run:"
    echo "   huggingface-cli login"
    echo "   Or set HF_TOKEN environment variable"
else
    echo "✅ HuggingFace authenticated"
fi

# Create training script
echo "📝 Creating training script..."
cat > start_training.sh << 'EOF'
#!/bin/bash
# Multi-GPU Training Script

# Configuration
MODEL="resnet50-pytorch"
DATASET="imagenet-1k"
DATA_DIR="/mnt/imagenet/dataset"
EPOCHS=90
BATCH_SIZE=512
GPUS=4  # Adjust based on your instance

# Activate environment
source /mnt/imagenet/resnet50-imagenet-1k/venv/bin/activate
cd /mnt/imagenet/resnet50-imagenet-1k

# Check if dataset exists
if [ ! -d "$DATA_DIR" ]; then
    echo "📥 Downloading ImageNet dataset..."
    python infra/dataset/download_imagenet.py
fi

# Start training
echo "🚀 Starting multi-GPU training..."
echo "   Model: $MODEL"
echo "   Dataset: $DATASET"
echo "   GPUs: $GPUS"
echo "   Batch size: $BATCH_SIZE"
echo "   Epochs: $EPOCHS"

torchrun --nproc_per_node=$GPUS train.py \
  --model $MODEL \
  --dataset $DATASET \
  --data-dir $DATA_DIR \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --multi-gpu \
  --sync-bn \
  --scheduler cosine \
  --augmentation strong \
  --hf-token $HF_TOKEN \
  --hf-repo your-username/imagenet-resnet50 \
  --resume-from ./checkpoint_1
EOF

chmod +x start_training.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > monitor.sh << 'EOF'
#!/bin/bash
# Training monitoring script

echo "🎮 GPU Status:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
while IFS=',' read -r name util mem_used mem_total temp; do
    printf "  %-20s | GPU: %3s%% | Memory: %6s/%6s MB | Temp: %3s°C\n" \
           "$name" "$util" "$mem_used" "$mem_total" "$temp"
done

echo ""
echo "💾 Disk Usage:"
df -h /mnt/imagenet

echo ""
echo "📈 Training Progress:"
if [ -f "training.log" ]; then
    tail -5 training.log
else
    echo "   No training.log found"
fi
EOF

chmod +x monitor.sh

# Create quick test script
echo "🧪 Creating test script..."
cat > test_setup.sh << 'EOF'
#!/bin/bash
# Quick test of multi-GPU setup

source /mnt/imagenet/resnet50-imagenet-1k/venv/bin/activate
cd /mnt/imagenet/resnet50-imagenet-1k

echo "🧪 Testing multi-GPU setup with ImageNette..."

torchrun --nproc_per_node=4 train.py \
  --model resnet50-pytorch \
  --dataset imagenet \
  --data-dir ./tiny-imagenet-200 \
  --num-classes 200 \
  --epochs 2 \
  --batch-size 128 \
  --multi-gpu \
  --sync-bn \
  --scheduler cosine \
  --augmentation none \
  --no-mixup \
  --visualize-samples
EOF

chmod +x test_setup.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Authenticate with HuggingFace:"
echo "   huggingface-cli login"
echo ""
echo "2. Download ImageNet dataset:"
echo "   python infra/dataset/download_imagenet.py"
echo ""
echo "3. Start training:"
echo "   ./start_training.sh"
echo ""
echo "4. Monitor training:"
echo "   ./monitor.sh"
echo ""
echo "5. Test setup (optional):"
echo "   ./test_setup.sh"
echo ""
echo "📊 Useful commands:"
echo "   watch -n 1 nvidia-smi          # Monitor GPUs"
echo "   tail -f training.log           # Watch training logs"
echo "   htop                          # Monitor system resources"
echo ""
echo "🎯 Ready for multi-GPU training!"
