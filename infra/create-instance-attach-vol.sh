# --- Required variables ---
AMI_ID="ami-0cc0caf8555831ce4"        # Replace with your AMI or custom AMI ID
KEY_NAME="erav4"               # Replace with your EC2 keypair
SG_ID="sg-256f3f6e"          # Replace with your security group
IAM_ROLE="erav4-ec2-role"  # Replace with your IAM instance profile
REGION="us-east-1"
AZ="us-east-1d"
INST_TYPE="g5.2xlarge"
#INST_TYPE="t3.micro"
VOL_ID="vol-029e937e578470342"


# Step 1: Launch instance and capture InstanceId
INSTANCE_ID=$(aws ec2 run-instances \
  --instance-type "$INST_TYPE" \
  --key-name "$KEY_NAME" \
  --placement "AvailabilityZone=$AZ" \
  --network-interfaces "{\"AssociatePublicIpAddress\":true,\"DeviceIndex\":0,\"Groups\":[\"$SG_ID\"]}" \
  --iam-instance-profile "{\"Arn\":\"arn:aws:iam::537907620791:instance-profile/$IAM_ROLE\"}" \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
  --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
  --count 1 \
  --region "$REGION" \
  --user-data 'IyEvYmluL2Jhc2gKc3VkbyBhcHQgdXBkYXRlIC15CnN1ZG8gYXB0IGluc3RhbGwgbWljcm8gZ3B1
c3RhdCAteQpzdWRvIG1rZGlyIC1wIC9tbnQvaW1hZ2VuZXQKCiMgV2FpdCB1cCB0byA1IG1pbnV0
ZXMgZm9yIC9kZXYvbnZtZTJuMSB0byBhcHBlYXIKbWF4X3dhaXQ9MzAwCmVsYXBzZWQ9MAp3aGls
ZSBbICRlbGFwc2VkIC1sdCAkbWF4X3dhaXQgXTsgZG8KICAgIGlmIFsgLWUgL2Rldi9udm1lMm4x
IF07IHRoZW4KICAgICAgICBlY2hvICJEZXZpY2UgL2Rldi9udm1lMm4xIGZvdW5kISIKICAgICAg
ICBzdWRvIG1vdW50IC9kZXYvbnZtZTJuMSAvbW50L2ltYWdlbmV0CiAgICAgICAgc3VkbyBjaG93
biB1YnVudHU6dWJ1bnR1IC9tbnQvaW1hZ2VuZXQKICAgICAgICBicmVhawogICAgZmkKICAgIGVj
aG8gIldhaXRpbmcgZm9yIGRldmljZS4uLiAoJGVsYXBzZWQgc2Vjb25kcykiCiAgICBzbGVlcCAx
MAogICAgZWxhcHNlZD0kKChlbGFwc2VkICsgMTApKQpkb25lCgojIEFkZCBlbnZpcm9ubWVudCBh
Y3RpdmF0aW9uIHRvIC5iYXNocmMKaWYgWyAtZiAvbW50L2ltYWdlbmV0L3ZlbnYvYmluL2FjdGl2
YXRlIF07IHRoZW4KICAgIGVjaG8gJ2lmIFsgLWYgL21udC9pbWFnZW5ldC92ZW52L2Jpbi9hY3Rp
dmF0ZSBdOyB0aGVuJyA+PiAvaG9tZS91YnVudHUvLmJhc2hyYwogICAgZWNobyAnICAgIHNvdXJj
ZSAvbW50L2ltYWdlbmV0L3ZlbnYvYmluL2FjdGl2YXRlJyA+PiAvaG9tZS91YnVudHUvLmJhc2hy
YwogICAgZWNobyAnZmknID4+IC9ob21lL3VidW50dS8uYmFzaHJjCiAgICBjaG93biB1YnVudHU6
dWJ1bnR1IC9ob21lL3VidW50dS8uYmFzaHJjCmZpCg==' \
  --image-id "$AMI_ID" \
  --tag-specifications '{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"resnet50-trainer"}, {"Key":"project","Value":"erav4"}]}' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Launched Instance: $INSTANCE_ID"

# Step 2: Wait until instance is running
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Instance $INSTANCE_ID is now running."

# Step 3: Attach the existing EBS volume
aws ec2 attach-volume \
  --volume-id "$VOL_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --device /dev/xvdf

echo "EBS volume attached successfully to $INSTANCE_ID"
echo "The user-data script will detect and mount the volume within 5 minutes."
