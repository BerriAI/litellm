# Rust Gateway Integration Guide

This guide explains how to integrate the high-performance Rust gateway with the LiteLLM Python proxy.

## Overview

The Rust gateway provides significantly better performance than the Python proxy:
- **10-15x higher throughput** (3,800 RPS vs 250 RPS)
- **Lower latency variance** under high concurrency
- **Smaller memory footprint** (22MB vs 200MB+)
- **Better resource efficiency** (no Python runtime overhead)

Two integration approaches are available, and both can be used simultaneously:

1. **PyO3 Bridge**: Direct function calls from Python to Rust
2. **Sidecar Gateway**: Rust gateway runs as separate process

## Quick Start

### Using the Deployment Script

The easiest way to deploy with Rust integration:

```bash
# Enable sidecar gateway
ENABLE_RUST_GATEWAY=true ./deploy_with_rust.sh

# Enable PyO3 bridge
LITELLM_RUST_PIPELINE=true ./deploy_with_rust.sh

# Enable both
ENABLE_RUST_GATEWAY=true LITELLM_RUST_PIPELINE=true ./deploy_with_rust.sh
```

### Manual Deployment

#### 1. Build the Rust Gateway

```bash
cd litellm-rust
cargo build --release --features server
```

The binary will be at `litellm-rust/target/release/litellm-ai-gateway`.

#### 2. Configure Environment Variables

```bash
# Sidecar gateway
export ENABLE_RUST_GATEWAY=true
export RUST_GATEWAY_PORT=4001
export RUST_GATEWAY_BINARY=litellm-rust/target/release/litellm-ai-gateway

# PyO3 bridge (requires compiled bridge)
export LITELLM_RUST_PIPELINE=true

# Standard LiteLLM config
export LITELLM_MASTER_KEY=sk-your-key
export DATABASE_URL=your-db-url
```

#### 3. Start the Proxy

```bash
python -m litellm --config config.yaml --port 4000
```

## Integration Modes

### PyO3 Bridge (Direct Calls)

**How it works:** Python calls Rust functions directly via PyO3.

**Pros:**
- Lowest latency (no HTTP overhead)
- Shared memory space
- Simpler deployment

**Cons:**
- Requires compiled PyO3 bridge
- Blocking calls (can't leverage Rust's async)
- Limited to routes implemented in Rust core

**Enable:** `LITELLM_RUST_PIPELINE=true`

**Check availability:**
```bash
python -c "from litellm.rust_bridge.loader import native_bridge_available; print(native_bridge_available())"
```

### Sidecar Gateway (Separate Process)

**How it works:** Rust gateway runs as separate process, Python forwards requests via HTTP.

**Pros:**
- Full Rust gateway features (circuit breaker, rate limiting, etc.)
- Independent scaling
- Better isolation

**Cons:**
- HTTP overhead between Python and Rust
- More complex deployment
- Requires health checking

**Enable:** `ENABLE_RUST_GATEWAY=true`

**Check health:**
```bash
curl http://localhost:4001/health/liveness
```

## Routing Logic

The proxy uses intelligent routing to decide which requests go to Rust:

1. **Sidecar gateway** (if enabled and healthy)
   - Routes: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
   - Skips: Streaming requests (for now)
   
2. **PyO3 bridge** (if enabled)
   - Routes: Same as sidecar
   - Falls back to Python if route not supported

3. **Python proxy** (fallback)
   - Handles all other requests
   - Handles streaming until Rust streaming is production-ready

## Configuration

Both integration modes use the same configuration files:

```yaml
# config.yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: sk-your-key
```

The Rust gateway reads the same config, so you don't need duplicate configuration.

## Monitoring

### Sidecar Gateway Metrics

```bash
# Prometheus metrics
curl http://localhost:4001/metrics

# Health check
curl http://localhost:4001/health/liveness
curl http://localhost:4001/health/readiness
curl http://localhost:4001/health/deep
```

### Logs

Both Python and Rust gateways log to stdout. Filter by source:

```bash
# Python proxy logs
python -m litellm ... 2>&1 | grep "LiteLLM"

# Rust gateway logs
# (automatically started by sidecar integration)
```

## Performance Tuning

### Sidecar Gateway

```bash
# Increase concurrent requests
export MAX_CONCURRENT_REQUESTS=2000

# Adjust rate limits
export GLOBAL_RATE_LIMIT=20000

# Configure timeouts
export REQUEST_TIMEOUT_SECS=60
```

### PyO3 Bridge

The PyO3 bridge runs in the Python process, so tuning is the same as the Python proxy:

```bash
# Increase workers
python -m litellm --workers 4 ...
```

## Troubleshooting

### Sidecar gateway not starting

**Check:**
```bash
# Is the binary built?
ls -la litellm-rust/target/release/litellm-ai-gateway

# Can it start manually?
./litellm-rust/target/release/litellm-ai-gateway

# Check logs
journalctl -u litellm -f
```

### PyO3 bridge not available

**Check:**
```bash
# Is the bridge compiled?
ls -la litellm/rust_bridge/_native.*

# Can it be imported?
python -c "from litellm.rust_bridge import _native"
```

### Requests not routing to Rust

**Check:**
```bash
# Is the gateway healthy?
curl http://localhost:4001/health/liveness

# Check proxy logs for routing decisions
python -m litellm ... 2>&1 | grep -i rust
```

## Migration Path

### Phase 1: Testing (Current)
- Enable Rust gateway in staging
- Monitor performance and errors
- Compare with Python-only mode

### Phase 2: Gradual Rollout
- Enable for specific models/endpoints
- Monitor error rates
- A/B test performance

### Phase 3: Full Production
- Enable for all traffic
- Monitor continuously
- Optimize configuration

## Architecture

```
┌─────────────────┐
│   Client        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Python Proxy (FastAPI)         │
│  - Auth, logging, callbacks     │
│  - Routing logic                │
└────────┬────────────────────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────┐
│ PyO3 Bridge     │  │ Sidecar     │  │ Python      │
│ (Direct calls)  │  │ Gateway     │  │ Fallback    │
│                 │  │ (HTTP)      │  │             │
└─────────────────┘  └──────┬──────┘  └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ Rust Core   │
                     │ - Routing   │
                     │ - Providers │
                     │ - Caching   │
                     └─────────────┘
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs from both Python and Rust
3. Open an issue on GitHub

## License

Same as LiteLLM - MIT License
