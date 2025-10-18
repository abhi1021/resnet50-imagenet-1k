# Create mount point and mount
sudo mkdir -p /mnt/imagenet
sudo mount /dev/nvme3n1 /mnt/imagenet

# Set ownership to ubuntu user
sudo chown ubuntu:ubuntu /mnt/imagenet

