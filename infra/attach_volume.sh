# Variables (replace as needed)
AZ="us-east-1f"
VOLUME_SIZE=400
INSTANCE_ID="i-09f870cb9f04e3e01"
DEVICE_NAME="/dev/xvdf"
VOLUME_ID="vol-029e937e578470342"

## Create 320GB EBS volume in required AZ
#VOLUME_ID=$(aws ec2 create-volume \
#  --size 400\
#  --availability-zone us-east-1f \
#  --volume-type gp3 \
#  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=imagenet-data}]' \
#  --query 'VolumeId' --output text)

# Attach to running EC2 instance
aws ec2 attach-volume \
  --volume-id "vol-029e937e578470342" \
  --instance-id "i-019175ebf204bf786" \
  --device "/dev/xvdf"

#{
#  "AttachTime": "2025-10-18T13:43:39.479000+00:00",
#  "Device": "/dev/xvdf",
#  "InstanceId": "i-02fbe56019aabe7f3",
#  "State": "attaching",
#  "VolumeId": "vol-029e937e578470342"
#}






