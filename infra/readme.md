
# Create a dowloadable readme for above.

Here’s a clear, ready-to-download README for preparing your AWS environment and downloading ImageNet-1k into a dedicated EBS volume. Copy this into a `README.md` file and share or archive as needed!

***

# ImageNet-1k Preparation and Download on AWS EC2

This README guides you through provisioning storage, launching a spot instance, attaching your EBS volume, and downloading ImageNet-1k using Hugging Face datasets.

***

## 1. **Create and Attach EBS Volume**

1. Create a volume in your AWS region (example: 400GB for ImageNet-1k):

```bash
aws ec2 create-volume \
  --size 400 \
  --availability-zone us-east-1a \
  --volume-type gp3 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=imagenet-data}]'
```

2. Attach to your running EC2 instance:

```bash
aws ec2 attach-volume \
  --volume-id <your-volume-id> \
  --instance-id <your-instance-id> \
  --device /dev/xvdf
```


***

## 2. **On Your EC2: Format and Mount the Volume**

SSH into your instance and run:

```bash
# Find the correct device, often /dev/nvme3n1 (use lsblk)
lsblk

# Format (once only!)
sudo mkfs.ext4 /dev/nvme3n1

# Mount it
sudo mkdir -p /mnt/imagenet
sudo mount /dev/nvme3n1 /mnt/imagenet
sudo chown ubuntu:ubuntu /mnt/imagenet
```


***

## 3. **Prepare Python Environment for Download**

To avoid system-wide pip issues, use a virtual environment:

```bash
cd /mnt/imagenet
python3 -m venv imagenet-venv
source imagenet-venv/bin/activate
pip install datasets huggingface_hub
```


***

## 4. **Download ImageNet-1k Dataset**

**With virtualenv activated:**

```bash
python -c 'from datasets import load_dataset; load_dataset("imagenet-1k", cache_dir="/mnt/imagenet")'
```

- This will download the dataset using Hugging Face’s datasets library and cache it to `/mnt/imagenet`.
- For large downloads, ensure you have enough disk space (>320GB recommended).

***

## 5. **Troubleshooting**

- Use `lsblk` to identify the correct block device name.
- If you see permission errors for `pip install`, always use a virtual environment.
- If you want to download the official ImageNet data, you must have approved credentials from image-net.org.

***

## 6. **Credits \& References**

- AWS Deep Learning AMI (Ubuntu)
- Hugging Face Datasets: [imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
- [PEP 668](https://peps.python.org/pep-0668/) (pip install best practices)

***

**Ready to train!** 🎉
Store, process, and reuse your EBS ImageNet-1k volume for future spot EC2 runs.

