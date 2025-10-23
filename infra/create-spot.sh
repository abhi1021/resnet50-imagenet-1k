# --- Required variables ---
AMI_ID="ami-03812fdd1e1001e52"        # Replace with your DLAMI or custom AMI ID
KEY_NAME="erav4"               # Replace with your EC2 keypair
SG_ID="sg-256f3f6e"          # Replace with your security group
SUBNET_ID="subnet-xxxxxxxxxxxxxxxxx"  # Replace with your subnet
IAM_ROLE="erav4-ec2-role"  # Replace with your IAM instance profile
REGION="us-east-1"
INST_TYPE="g4dn.xlarge"


aws ec2 describe-spot-price-history \
    --instance-types g5.2xlarge \
    --start-time "$(python3 -c 'from datetime import datetime, timedelta; print((datetime.utcnow() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"))')" \
    --product-descriptions "Linux/UNIX" \
    --region us-east-1

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

aws service-quotas list-service-quotas --service-code ec2 --region sa-east-1 \
  --query "Quotas[?contains(Name, 'Spot')].{Name:Name, Value:Value, Adjustable:Adjustable, QuotaCode:QuotaCode}" \
  --output table

#
## -- Block device mapping (16GB root only) --
#BLOCK_MAP='[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":16,"VolumeType":"gp3","DeleteOnTermination":true}}]'

  INSTANCE_ID=$(aws ec2 run-instances \
  --instance-type 'g5.2xlarge' \
  --key-name 'erav4' \
  --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-b77becd2"]}' \
  --iam-instance-profile '{"Arn":"arn:aws:iam::537907620791:instance-profile/erav4-ec2-role"}' \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.25","SpotInstanceType":"one-time"}}' \
  --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
  --placement '{"AvailabilityZone":"sa-east-1a"}' \
  --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}'\
  --count '1' \
  --region sa-east-1 \
  --user-data 'c3VkbyBta2RpciAtcCAvbW50L2ltYWdlbmV0CnN1ZG8gbW91bnQgL2Rldi9udm1lMm4xIC9tbnQvaW1hZ2VuZXQKc3VkbyBjaG93biB1YnVudHU6dWJ1bnR1IC9tbnQvaW1hZ2VuZXQ=' \
  --image-id 'ami-05c34bd1d228e89b0' \
  --tag-specifications '{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"resnet50-trainer"}, {"Key":"project","Value":"erav4"}]}'\
  --query 'InstanceId' --output text)
