# --- Required variables ---
AMI_ID="ami-0b6ee6f9948ada413"        # Replace with your DLAMI or custom AMI ID
KEY_NAME="erav4"               # Replace with your EC2 keypair
SG_ID="sg-256f3f6e"          # Replace with your security group
SUBNET_ID="subnet-xxxxxxxxxxxxxxxxx"  # Replace with your subnet
IAM_ROLE="erav4-ec2-role"  # Replace with your IAM instance profile
REGION="us-east-1"

# -- Block device mapping (16GB root only) --
BLOCK_MAP='[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":16,"VolumeType":"gp3","DeleteOnTermination":true}}]'

# --- Spot instance request ---
aws ec2 run-instances \
  --region $REGION \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
  --instance-type g4dn.xlarge \
  --image-id $AMI_ID \
  --key-name $KEY_NAME \
  --security-group-ids $SG_ID \
  --block-device-mappings "$BLOCK_MAP" \
  --iam-instance-profile Name=$IAM_ROLE \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=imagenet-spot}]'