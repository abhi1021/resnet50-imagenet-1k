#### ⚠️ ONE TIME TASK ONLY ####
### CREATE BUCKET ####
#aws s3api create-bucket --bucket erav4-imagenet-training --region eu-west-2

# Create the IAM role for EC2
aws iam create-role \
  --role-name erav4-ec2-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'\
  --region eu-west-2

# Attach your custom policy as inline policy
aws iam put-role-policy \
  --role-name erav4-ec2-role \
  --policy-name s3-single-bucket-access-policy \
  --policy-document file://s3-single-bucket-access.json
# Create instance profile
aws iam create-instance-profile --instance-profile-name erav4-ec2-role

# Add your role to the profile
aws iam add-role-to-instance-profile --instance-profile-name erav4-ec2-role --role-name erav4-ec2-role