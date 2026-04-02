#!/bin/bash
set -e

IMAGE_NAME="pam-litellm"

if [[ "$(docker images -q $IMAGE_NAME 2>/dev/null)" == "" ]]; then
  echo "Image not found, building..."
  docker build --platform linux/amd64 -t $IMAGE_NAME .
fi

echo "Starting LiteLLM on http://localhost:4000"
docker run --rm \
  -p 4000:4000 \
  -e GEMINI_API_KEY="${GEMINI_API_KEY:?GEMINI_API_KEY is not set}" \
  -e LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY is not set}" \
  "$IMAGE_NAME"
