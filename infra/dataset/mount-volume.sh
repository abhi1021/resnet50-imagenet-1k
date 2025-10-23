
# lsblk
# Use above command and find correct mount type eg: below one was /dev/nvme2n1

# Create mount point and mount
sudo mkdir -p /mnt/imagenet
sudo mount /dev/nvme2n1 /mnt/imagenet
sudo chown ubuntu:ubuntu /mnt/imagenet

cd /mnt/imagenet
source venv/bin/activate
cd resnet50-imagenet-1k

# Start the training, refer to the command in the message. Here it's for reference only
python train.py --model resnet50-pytorch --dataset imagenet-1k --data-dir /mnt/imagenet/dataset --epochs 100 --batch-size 256 --scheduler onecycle --lr-finder --resume-from ./checkpoint_3 --hf-token <hf token> --hf-repo <hf repo> >> console-3.log  2>&1 &

# watch GPU usage
watch -n 0.01 nvidia-smi

# monitor CPU and memory usage
htop

# To check the progress of the training, the console output is redirected to the console-3.log file.
tail -f console-3.log 
