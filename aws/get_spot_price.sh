#!/usr/bin/env bash

# All P2 Instances
aws ec2 describe-spot-price-history --instance-types p2.xlarge --product-description Linux/UNIX --page-size 2 --max-items 12
# aws ec2 describe-spot-price-history --instance-types p2.8xlarge --product-description Linux/UNIX --page-size 2 --max-items 1
#aws ec2 describe-spot-price-history --instance-types p2.16xlarge --product-description Linux/UNIX --page-size 2 --max-items 3

# All P3 Instances
# aws ec2 describe-spot-price-history --instance-types p3.2xlarge --product-description Linux/UNIX --page-size 2 --max-items 3
#aws ec2 describe-spot-price-history --instance-types p3.8xlarge --product-description Linux/UNIX --page-size 2 --max-items 3
#aws ec2 describe-spot-price-history --instance-types p3.16xlarge --product-description Linux/UNIX --page-size 2 --max-items 3

# ./get_spot_price.sh | sed -n 'p;n'
