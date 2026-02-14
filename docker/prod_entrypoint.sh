#!/bin/sh

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

if [ "$SEPARATE_HEALTH_APP" = "1" ]; then
    export LITELLM_ARGS="$@"
    export SUPERVISORD_STOPWAITSECS="${SUPERVISORD_STOPWAITSECS:-3600}"
    exec supervisord -c /etc/supervisord.conf
fi

if [ "$USE_DDTRACE" = "true" ]; then
    export DD_TRACE_OPENAI_ENABLED="False"
    exec ddtrace-run litellm "$@"
else
    exec litellm "$@"
fi