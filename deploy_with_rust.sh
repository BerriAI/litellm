#!/bin/bash
# Deployment script for LiteLLM with Rust gateway integration
#
# This script starts the Python proxy with optional Rust gateway integration.
# Two integration approaches are available:
#
# 1. PyO3 Bridge (direct function calls):
#    Set LITELLM_RUST_PIPELINE=true
#    Requires: Compiled PyO3 bridge in litellm/rust_bridge/_native.*
#
# 2. Sidecar Gateway (separate process):
#    Set ENABLE_RUST_GATEWAY=true
#    Requires: Compiled Rust gateway binary
#
# Both can be enabled simultaneously for maximum performance.

set -e

# Configuration
LITELLM_CONFIG=${LITELLM_CONFIG:-"config.yaml"}
LITELLM_PORT=${LITELLM_PORT:-"4000"}
LITELLM_HOST=${LITELLM_HOST:-"0.0.0.0"}
LITELLM_WORKERS=${LITELLM_WORKERS:-"1"}

# Rust gateway configuration
RUST_GATEWAY_PORT=${RUST_GATEWAY_PORT:-"4001"}
RUST_GATEWAY_BINARY=${RUST_GATEWAY_BINARY:-"litellm-rust/target/release/litellm-ai-gateway"}

# Shared state configuration (both Python and Rust use same Redis/Postgres)
REDIS_URL=${REDIS_URL:-""}
DATABASE_URL=${DATABASE_URL:-""}

# Integration mode
ENABLE_RUST_GATEWAY=${ENABLE_RUST_GATEWAY:-"false"}
LITELLM_RUST_PIPELINE=${LITELLM_RUST_PIPELINE:-"false"}

echo "=========================================="
echo "LiteLLM Proxy with Rust Gateway"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Config file: $LITELLM_CONFIG"
echo "  Host: $LITELLM_HOST"
echo "  Port: $LITELLM_PORT"
echo "  Workers: $LITELLM_WORKERS"
echo ""
echo "Rust Integration:"
echo "  Sidecar gateway: $ENABLE_RUST_GATEWAY"
echo "  PyO3 bridge: $LITELLM_RUST_PIPELINE"
echo ""

# Check if config file exists
if [ ! -f "$LITELLM_CONFIG" ]; then
    echo "Error: Config file not found: $LITELLM_CONFIG"
    echo "Please create a config.yaml or set LITELLM_CONFIG environment variable"
    exit 1
fi

# Export environment variables
export LITELLM_CONFIG
export LITELLM_PORT
export LITELLM_HOST
export ENABLE_RUST_GATEWAY
export LITELLM_RUST_PIPELINE
export REDIS_URL
export DATABASE_URL

if [ "$ENABLE_RUST_GATEWAY" = "true" ]; then
    export RUST_GATEWAY_PORT
    export RUST_GATEWAY_BINARY
    
    # Check if Rust gateway binary exists
    if [ ! -f "$RUST_GATEWAY_BINARY" ]; then
        echo "Warning: Rust gateway binary not found at $RUST_GATEWAY_BINARY"
        echo "Building Rust gateway..."
        cd litellm-rust
        cargo build --release --features server
        cd ..
    fi
    
    echo "✓ Rust sidecar gateway enabled on port $RUST_GATEWAY_PORT"
fi

if [ "$LITELLM_RUST_PIPELINE" = "true" ]; then
    # Check if PyO3 bridge is available
    if python -c "from litellm.rust_bridge.loader import native_bridge_available; exit(0 if native_bridge_available() else 1)" 2>/dev/null; then
        echo "✓ PyO3 Rust bridge enabled"
    else
        echo "Warning: PyO3 bridge not available"
        echo "The compiled bridge (litellm/rust_bridge/_native.*) is missing"
        echo "Falling back to Python-only mode"
        export LITELLM_RUST_PIPELINE="false"
    fi
fi

echo ""
echo "Starting LiteLLM Proxy Server..."
echo ""

# Start the proxy
exec python -m litellm \
    --config "$LITELLM_CONFIG" \
    --host "$LITELLM_HOST" \
    --port "$LITELLM_PORT" \
    --workers "$LITELLM_WORKERS"
