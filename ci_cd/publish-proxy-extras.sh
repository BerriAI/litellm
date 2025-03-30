#!/bin/bash

# Exit on error
set -e

echo "🚀 Building and publishing litellm-proxy-extras"

# Navigate to litellm-proxy-extras directory
cd "$(dirname "$0")/../litellm-proxy-extras"

# Build the package
echo "📦 Building package..."
poetry build

# Publish to PyPI
echo "🌎 Publishing to PyPI..."
poetry publish

echo "✅ Done! Package published successfully"