
# lsblk
# Use above command and find correct mount type eg: below one was /dev/nvme2n1

# Create mount point and mount
sudo mkdir -p /mnt/imagenet
sudo mount /dev/nvme2n1 /mnt/imagenet
sudo chown ubuntu:ubuntu /mnt/imagenet

