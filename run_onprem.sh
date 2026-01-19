#!/bin/bash

# Script to run LiteLLM on-prem version with all enterprise features unlocked

set -e  # Exit on any error

echo "Starting LiteLLM on-prem server with enterprise features..."
echo "========================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if the litellm-onprem image exists
if ! docker images | grep -q "litellm-onprem"; then
    echo "⚠️  LiteLLM on-prem image not found. Building it first..."
    ./build_onprem.sh
fi

# Run the server
echo "🚀 Starting LiteLLM on-prem server..."
echo "Server will be available at http://localhost:4000"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the container with port mapping
docker run -p 4000:4000 litellm-onprem "$@"

echo ""
echo "Server stopped."
