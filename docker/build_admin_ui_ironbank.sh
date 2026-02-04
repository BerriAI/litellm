#!/bin/bash

# Admin UI build script for Iron Bank / UBI-based images
# Uses dnf instead of apt-get/apk

echo "Current directory:"
pwd

# Only run this step for litellm enterprise
if [ ! -f "enterprise/enterprise_ui/enterprise_colors.json" ]; then
    echo "Admin UI - using default LiteLLM UI"
    exit 0
fi

echo "Building Custom Admin UI..."

# Install curl using dnf (for RHEL/UBI)
if command -v dnf &> /dev/null; then
    dnf install -y curl nodejs npm
elif command -v microdnf &> /dev/null; then
    microdnf install -y curl nodejs npm
else
    echo "Error: No supported package manager found (dnf/microdnf)"
    exit 1
fi

# Install nvm and Node.js v18
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
source ~/.nvm/nvm.sh
nvm install v18.17.0
nvm use v18.17.0
npm install -g npm

# Copy enterprise colors config
cp enterprise/enterprise_ui/enterprise_colors.json ui/litellm-dashboard/ui_colors.json

# Build the UI
cd ui/litellm-dashboard
chmod +x ./build_ui.sh
./build_ui.sh

# Return to root directory
cd ../..
