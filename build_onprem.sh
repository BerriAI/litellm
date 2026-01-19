#!/bin/bash

# Script to build on-prem version of LiteLLM with all enterprise features unlocked

set -e  # Exit on any error

echo "Building on-prem version of LiteLLM with enterprise features..."

# Navigate to the project root directory
cd "$(dirname "$0")"

# Build the on-prem Docker image
echo "Building Docker image..."
docker build -t litellm-onprem -f docker/build_from_pip/Dockerfile.onprem .

echo "Build completed successfully!"
echo "To run the on-prem version:"
echo "  docker run -p 4000:4000 litellm-onprem"
echo ""
echo "All enterprise features are now permanently unlocked without any license requirements."
