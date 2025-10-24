#!/bin/bash
# Update and install required packages
sudo apt update -y
sudo apt install micro gpustat -y

# Create mount point
sudo mkdir -p /mnt/imagenet

# Retry mounting the EBS volume up to 5 times with 10-second delay
max_attempts=5
attempt=1
while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt: Trying to mount /dev/nvme2n1..."
    sudo mount /dev/nvme2n1 /mnt/imagenet && break
    echo "Mount failed, retrying in 10 seconds..."
    sleep 10
    attempt=$((attempt + 1))
done

# Final check
if mountpoint -q /mnt/imagenet; then
    echo "Mount successful!"
else
    echo "Mount failed after $max_attempts attempts. Exiting."
    exit 1
fi

# Set correct ownership
sudo chown ubuntu:ubuntu /mnt/imagenet

# Optional: activate environment if mount successful
if [ -f /mnt/imagenet/venv/bin/activate ]; then
    source /mnt/imagenet/venv/bin/activate
fi

# Auto-activate environment on future logins
sudo bash -c 'echo "if [ -f /mnt/imagenet/venv/bin/activate ]; then" >> /home/ubuntu/.bashrc'
sudo bash -c 'echo "    source /mnt/imagenet/venv/bin/activate" >> /home/ubuntu/.bashrc'
sudo bash -c 'echo "fi" >> /home/ubuntu/.bashrc'

# Ensure .bashrc ownership
sudo chown ubuntu:ubuntu /home/ubuntu/.bashrc
