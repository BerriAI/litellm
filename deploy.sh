#!/bin/bash
set -e


LITELLM_IMAGE="199658938451.dkr.ecr.us-east-2.amazonaws.com/litellm"
IMAGE_NAME="$LITELLM_IMAGE:latest"


echo "Building image $IMAGE_NAME"

docker buildx create --use --name=crossplat --node=crossplat

docker buildx build \
    --push \
    --platform linux/amd64,linux/arm64 \
    --cache-to mode=max,image-manifest=true,oci-mediatypes=true,type=registry,ref=$LITELLM_IMAGE:cache,ignore-error=true,timeout=2m \
    --cache-from type=registry,ref=$LITELLM_IMAGE:cache \
    --tag "$IMAGE_NAME" \
    --file docker/Dockerfile.database .
