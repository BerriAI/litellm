#!/bin/bash
echo $(pwd)

# Copy pre-cached Prisma binaries to writable location for read-only filesystem environments (AKS, etc.)
if [ -d "/app/.cache/prisma-python/binaries" ] && [ ! -d "/tmp/.cache/prisma-python/binaries" ]; then
    echo "Copying pre-cached Prisma binaries to /tmp for read-only filesystem compatibility..."
    mkdir -p /tmp/.cache/prisma-python
    cp -r /app/.cache/prisma-python/binaries /tmp/.cache/prisma-python/
    echo "Prisma binaries copied successfully"
fi

# Set runtime cache locations to /tmp (writable even in read-only root filesystem)
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
export NPM_CONFIG_CACHE="${NPM_CONFIG_CACHE:-/tmp/npm-cache}"
export PRISMA_BINARY_CACHE_DIR="${PRISMA_BINARY_CACHE_DIR:-/tmp/.cache/prisma-python/binaries}"

# Run the Python migration script
python3 litellm/proxy/prisma_migration.py

# Check if the Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Migration script ran successfully!"
else
    echo "Migration script failed!"
    exit 1
fi
