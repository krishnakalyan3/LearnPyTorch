#!/usr/bin/env bash

SPOT_PRICE=$1
aws ec2 request-spot-instances \
	--cli-input-json=file://spec.json \
	--spot-price $SPOT_PRICE --region us-east-1