#!/bin/bash
set -euo pipefail

# Update OS and install dos2unix
sudo yum update -y
sudo yum install -y dos2unix

# Fix all your sh scripts
# Fix all your .sh scripts if any exist
for f in /home/ec2-user/asr-fastapi-app/scripts/*.sh; do
  [ -f "$f" ] && sudo dos2unix "$f"
done

# Install Nginx
sudo yum install -y nginx
