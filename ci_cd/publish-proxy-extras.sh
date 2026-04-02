#!/bin/bash

# Exit on error
set -e

echo "🚀 Building and publishing litellm-proxy-extras"

# Navigate to litellm-proxy-extras directory
cd "$(dirname "$0")/../litellm-proxy-extras"

# Build the package
echo "📦 Building package..."
uv build

# Publish to PyPI
echo "🌎 Publishing to PyPI..."
uv publish

echo "✅ Done! Package published successfully"
