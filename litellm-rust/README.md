# LiteLLM Rust Gateway

High-performance, production-ready Rust implementation of the LiteLLM proxy. Provides a unified OpenAI-compatible interface to multiple LLM providers with enterprise-grade features.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, AWS Bedrock, and more
- **High Performance**: 3-10x faster than Python proxy
- **Production Hardening**: Circuit breakers, retry logic, rate limiting
- **Enterprise Features**: Spend tracking, audit logging, metrics
- **OpenAI Compatible**: Drop-in replacement for OpenAI API

## Quick Start

```bash
# Build
cargo build --release --features server

# Run
LITELLM_YAML_CONFIG=config.yaml ./target/release/litellm-ai-gateway

# Test
curl http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Documentation

- **[User Guide](USER_GUIDE.md)** - Configuration, usage, and features
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment (standalone, Docker, Kubernetes)
- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference
- **[Benchmark Guide](BENCHMARK_GUIDE.md)** - Performance testing and benchmarks
- **[Project Summary](PROJECT_SUMMARY.md)** - Project overview and architecture

## Architecture

```
crates/
  core/              # Core SDK: provider transforms, routing, types
  ai-gateway/        # Axum server with production hardening
  python-bridge/     # PyO3 bridge for Python integration
```

## Performance

The Rust gateway provides significant performance improvements over the Python proxy:

- **Latency**: 2-5x lower latency
- **Throughput**: 3-10x higher RPS
- **Resources**: Lower CPU and memory usage
- **Concurrency**: Better handling of concurrent requests

See [Benchmark Guide](BENCHMARK_GUIDE.md) for detailed benchmarks.

## Production Features

- **Circuit Breaker**: Automatic failure detection and recovery
- **Retry Logic**: Exponential backoff with jitter
- **Rate Limiting**: Per-key RPM, TPM, and parallel request limits
- **Spend Tracking**: Real-time spend tracking with Redis/PostgreSQL
- **Metrics**: Prometheus metrics for monitoring
- **Health Checks**: Liveness, readiness, and deep health checks
- **Audit Logging**: Comprehensive audit trail
- **TLS Support**: TLS termination support

## Configuration

See [User Guide](USER_GUIDE.md) for detailed configuration options.

Basic example:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: ${OPENAI_API_KEY}

general_settings:
  master_key: ${LITELLM_MASTER_KEY}
  redis_url: ${REDIS_URL}
  database_url: ${DATABASE_URL}
```

## Deployment

See [Deployment Guide](DEPLOYMENT_GUIDE.md) for detailed deployment instructions.

### Standalone
```bash
./target/release/litellm-ai-gateway
```

### Docker
```bash
docker build -t litellm-gateway .
docker run -p 4001:4001 litellm-gateway
```

### Kubernetes
See [Deployment Guide](DEPLOYMENT_GUIDE.md#kubernetes-deployment) for Kubernetes manifests.

## Testing

```bash
# Run all tests
cargo test --workspace

# Run with features
cargo test --features server

# Run benchmarks
cd bench && python compare_benchmark.py
```

## Contributing

See the main LiteLLM repository for contribution guidelines.

## License

See the main LiteLLM repository for license information.
