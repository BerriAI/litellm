#!/bin/bash
echo $(pwd)

# Copy pre-cached Prisma binaries to writable location for read-only filesystem environments (AKS, etc.)
if [ -d "/app/.cache/prisma-python/binaries" ] && [ ! -d "/tmp/.cache/prisma-python/binaries" ]; then
    echo "Copying pre-cached Prisma binaries to /tmp for read-only filesystem compatibility..."
    mkdir -p /tmp/.cache/prisma-python
    cp -r /app/.cache/prisma-python/binaries /tmp/.cache/prisma-python/
    echo "Prisma binaries copied successfully"
fi

# Override Dockerfile ENV defaults that point to read-only /app/.cache,
# but respect user-provided overrides (e.g. via K8s pod spec, docker-compose).
# The Dockerfile.non_root sets these to /app/.cache/... at build time, but at
# runtime /app is read-only, so we redirect to /tmp which is always writable.
if [ "$XDG_CACHE_HOME" = "/app/.cache" ] || [ -z "$XDG_CACHE_HOME" ]; then
    export XDG_CACHE_HOME="/tmp/.cache"
fi
if [ "$NPM_CONFIG_CACHE" = "/app/.cache/npm" ] || [ -z "$NPM_CONFIG_CACHE" ]; then
    export NPM_CONFIG_CACHE="/tmp/npm-cache"
fi
if [ "$PRISMA_BINARY_CACHE_DIR" = "/app/.cache/prisma-python/binaries" ] || [ -z "$PRISMA_BINARY_CACHE_DIR" ]; then
    export PRISMA_BINARY_CACHE_DIR="/tmp/.cache/prisma-python/binaries"
fi

# Run the Python migration script
python3 litellm/proxy/prisma_migration.py

# Check if the Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Migration script ran successfully!"
else
    echo "Migration script failed!"
    exit 1
fi
