#!/bin/bash
set -e

# Configuration
ECR_REPO="420049223852.dkr.ecr.eu-central-1.amazonaws.com/litellm-deepkeep"
TAG="${1:-dev-$(date +%Y%m%d-%H%M%S)}"
FULL_IMAGE="${ECR_REPO}:${TAG}"

echo "================================================"
echo "Building LiteLLM Docker image with UI"
echo "Image: ${FULL_IMAGE}"
echo "================================================"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 420049223852.dkr.ecr.eu-central-1.amazonaws.com

# Build the image
echo "Building Docker image..."
docker build -t "${FULL_IMAGE}" -f Dockerfile .

# Push the image
echo "Pushing image to ECR..."
docker push "${FULL_IMAGE}"

echo "================================================"
echo "✅ Image pushed successfully!"
echo ""
echo "To use in Tilt, update your tilt_config.yaniv-v2.yaml:"
echo ""
echo "litellm:"
echo "  image:"
echo "    repository: ${ECR_REPO}"
echo "    tag: ${TAG}"
echo ""
echo "Or run: tilt trigger litellm"
echo "================================================"
