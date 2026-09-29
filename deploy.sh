#!/bin/bash
set -e


LITELLM_IMAGE="199658938451.dkr.ecr.us-east-2.amazonaws.com/litellm"
BASE_IMAGE="$LITELLM_IMAGE:base"
FINAL_IMAGE="$LITELLM_IMAGE:basewmcp"

./buildmcp.sh

echo "Building image $BASE_IMAGE"

docker buildx create --use --name=crossplat --node=crossplat

docker buildx build \
    --push \
    --platform linux/amd64,linux/arm64 \
    --cache-to mode=max,image-manifest=true,oci-mediatypes=true,type=registry,ref=$LITELLM_IMAGE:cache,ignore-error=true,timeout=2m \
    --cache-from type=registry,ref=$LITELLM_IMAGE:cache \
    --tag "$BASE_IMAGE" \
    --file docker/Dockerfile.database .

echo "Building image $FINAL_IMAGE"
docker buildx build \
    --build-arg "LITELLM_BASE_IMAGE=$BASE_IMAGE" \
    --platform linux/amd64,linux/arm64 \
    --tag "$FINAL_IMAGE" \
    --file aven.Dockerfile \
    --push .
echo "Done building image $FINAL_IMAGE"
