# --- Required variables ---
AMI_ID="ami-03812fdd1e1001e52"        # Replace with your DLAMI or custom AMI ID
KEY_NAME="erav4"               # Replace with your EC2 keypair
SG_ID="sg-256f3f6e"          # Replace with your security group
SUBNET_ID="subnet-xxxxxxxxxxxxxxxxx"  # Replace with your subnet
IAM_ROLE="erav4-ec2-role"  # Replace with your IAM instance profile
REGION="us-east-1"
INST_TYPE="g4dn.xlarge"

#
## -- Block device mapping (16GB root only) --
#BLOCK_MAP='[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":16,"VolumeType":"gp3","DeleteOnTermination":true}}]'

  INSTANCE_ID=$(aws ec2 run-instances \
  --instance-type 'g5.2xlarge' \
  --key-name 'erav4' \
  --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-256f3f6e"]}' \
  --iam-instance-profile '{"Arn":"arn:aws:iam::537907620791:instance-profile/erav4-ec2-role"}' \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.44","SpotInstanceType":"one-time"}}' \
  --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
  --placement '{"AvailabilityZone":"us-east-1f"}' \
  --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}'\
  --count '1' \
  --user-data 'c3VkbyBta2RpciAtcCAvbW50L2ltYWdlbmV0CnN1ZG8gbW91bnQgL2Rldi9udm1lMm4xIC9tbnQvaW1hZ2VuZXQKc3VkbyBjaG93biB1YnVudHU6dWJ1bnR1IC9tbnQvaW1hZ2VuZXQ=' \
  --image-id 'ami-082cfdbb3062d6871' \
  --tag-specifications '{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"resnet50-trainer"}, {"Key":"project","Value":"erav4"}]}'\
  --query 'InstanceId' --output text)

