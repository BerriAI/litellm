# LiteLLM Rust Gateway - User Guide

## Introduction

The LiteLLM Rust Gateway is a high-performance, production-ready proxy for LLM APIs. It provides a unified OpenAI-compatible interface to multiple LLM providers with enterprise-grade features including rate limiting, circuit breakers, retry logic, spend tracking, and more.

## Quick Start

### Prerequisites

- Rust 1.70 or later
- Access to at least one LLM provider API key (OpenAI, Anthropic, etc.)
- (Optional) Redis for rate limiting and spend tracking
- (Optional) PostgreSQL for persistent spend logs

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm/litellm-rust
```

2. Build the gateway:
```bash
cargo build --release --features server
```

The binary will be at `target/release/litellm-ai-gateway`.

### Configuration

Create a configuration file `config.yaml`:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: sk-your-openai-key
  
  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: sk-ant-your-anthropic-key

general_settings:
  master_key: sk-your-master-key
  
  # Optional: Redis for rate limiting
  redis_url: redis://localhost:6379
  
  # Optional: PostgreSQL for spend logs
  database_url: postgresql://user:pass@localhost:5432/litellm
```

### Running the Gateway

```bash
# Basic usage
LITELLM_YAML_CONFIG=config.yaml ./target/release/litellm-ai-gateway

# With environment variables
LITELLM_YAML_CONFIG=config.yaml \
LITELLM_MASTER_KEY=sk-your-master-key \
REDIS_URL=redis://localhost:6379 \
DATABASE_URL=postgresql://user:pass@localhost:5432/litellm \
./target/release/litellm-ai-gateway
```

The gateway will start on `http://127.0.0.1:4001` by default.

### Making Your First Request

```bash
curl http://127.0.0.1:4001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Configuration Guide

### Model List

The `model_list` section defines which models are available and how to route to them.

```yaml
model_list:
  - model_name: gpt-4              # Name users will use in requests
    litellm_params:
      model: openai/gpt-4          # Provider/model format
      api_key: sk-...              # API key for this provider
      api_base: https://...        # Optional: custom API endpoint
      
  - model_name: my-clause          # Alias for Claude
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: sk-ant-...
```

### General Settings

```yaml
general_settings:
  # Master key for gateway authentication
  master_key: sk-your-master-key
  
  # Redis for rate limiting and caching
  redis_url: redis://localhost:6379
  
  # PostgreSQL for persistent storage
  database_url: postgresql://user:pass@localhost:5432/litellm
  
  # Request timeout in seconds (default: 600)
  request_timeout: 600
  
  # Cache TTL in seconds (default: 300)
  cache_ttl: 300
```

## Features

### Authentication

The gateway supports two authentication methods:

1. **Master Key**: A single key for gateway access (set in `master_key`)
2. **Per-Key Auth**: Individual API keys with their own limits and permissions (stored in database)

All requests must include the Authorization header:
```bash
-H "Authorization: Bearer YOUR_API_KEY"
```

### Rate Limiting

When Redis is configured, the gateway enforces rate limits per API key:

- **RPM** (Requests Per Minute)
- **TPM** (Tokens Per Minute)
- **Max Parallel Requests**

Rate limits are configured per API key in the database.

### Circuit Breaker

The circuit breaker protects against cascading failures when a provider is down:

- **Closed**: Normal operation, requests pass through
- **Open**: Provider is failing, requests fail fast
- **Half-Open**: Testing if provider has recovered

The circuit breaker automatically transitions between states based on failure rates.

### Retry Logic

Failed requests are automatically retried with exponential backoff:

- Maximum 3 retry attempts
- Exponential backoff with jitter
- Only retries on transient failures (network errors, timeouts)

### Spend Tracking

The gateway tracks spend for each API call:

- Real-time spend tracking via Redis
- Persistent spend logs in PostgreSQL
- Per-key, per-user, per-team, per-org spend tracking

### Streaming Support

The gateway supports streaming responses via Server-Sent Events (SSE):

```bash
curl http://127.0.0.1:4001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Health Checks

The gateway provides health check endpoints:

- `GET /health/liveness` - Is the gateway running?
- `GET /health/readiness` - Is the gateway ready to accept requests?
- `GET /health/deep` - Deep health check (checks Redis, PostgreSQL, providers)

### Metrics

Prometheus metrics are available at `GET /metrics`:

- Request counts and latency
- Token usage
- Spend tracking
- Circuit breaker state
- Rate limit statistics

## API Reference

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

**Request Body:**
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### POST /v1/messages

Anthropic Messages API endpoint.

**Request Body:**
```json
{
  "model": "claude-3-opus-20240229",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100
}
```

### POST /v1/embeddings

OpenAI-compatible embeddings endpoint.

**Request Body:**
```json
{
  "model": "text-embedding-3-small",
  "input": "The text to embed"
}
```

### POST /v1/images/generations

OpenAI-compatible image generation endpoint.

**Request Body:**
```json
{
  "model": "dall-e-3",
  "prompt": "A cute cat",
  "size": "1024x1024"
}
```

## Monitoring

### Logs

The gateway outputs structured JSON logs to stdout. Set the log level with `RUST_LOG`:

```bash
RUST_LOG=info ./target/release/litellm-ai-gateway
```

Log levels: `error`, `warn`, `info`, `debug`, `trace`

### Metrics

Prometheus metrics are available at `http://127.0.0.1:4001/metrics`.

Key metrics:
- `litellm_requests_total` - Total request count
- `litellm_request_duration_seconds` - Request latency histogram
- `litellm_tokens_total` - Total tokens processed
- `litellm_spend_usd_total` - Total spend in USD

### Health Checks

Use the health check endpoints for monitoring:

```bash
# Liveness check (is the process running?)
curl http://127.0.0.1:4001/health/liveness

# Readiness check (is the gateway ready?)
curl http://127.0.0.1:4001/health/readiness

# Deep health check (checks all dependencies)
curl http://127.0.0.1:4001/health/deep
```

## Troubleshooting

### Gateway won't start

1. Check that the config file exists and is valid YAML
2. Check that all required environment variables are set
3. Check the logs for error messages

### Requests are failing

1. Check the logs for error messages
2. Verify the API key is valid and has permissions
3. Check that the provider is accessible
4. Check the circuit breaker state via metrics

### High latency

1. Check the provider latency
2. Check network connectivity
3. Check if the circuit breaker is open
4. Check Redis/PostgreSQL performance

### Rate limit errors

1. Check the rate limits configured for the API key
2. Check Redis for current rate limit counters
3. Consider increasing rate limits or using a different API key

## Production Deployment

### System Requirements

- **CPU**: 2+ cores recommended
- **Memory**: 512MB+ recommended
- **Disk**: 100MB+ for binary and logs
- **Network**: Low latency to LLM providers

### Deployment Options

1. **Binary**: Deploy the compiled binary directly
2. **Docker**: Create a Docker image (see Docker section below)
3. **Kubernetes**: Deploy as a Kubernetes deployment

### Docker

Create a `Dockerfile`:

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --features server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/litellm-ai-gateway /usr/local/bin/
EXPOSE 4001
CMD ["litellm-ai-gateway"]
```

Build and run:

```bash
docker build -t litellm-gateway .
docker run -p 4001:4001 \
  -e LITELLM_YAML_CONFIG=/config/config.yaml \
  -v $(pwd)/config.yaml:/config/config.yaml \
  litellm-gateway
```

### Kubernetes

Create a Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: litellm-gateway
  template:
    metadata:
      labels:
        app: litellm-gateway
    spec:
      containers:
      - name: gateway
        image: litellm-gateway:latest
        ports:
        - containerPort: 4001
        env:
        - name: LITELLM_YAML_CONFIG
          value: /config/config.yaml
        - name: LITELLM_MASTER_KEY
          valueFrom:
            secretKeyRef:
              name: litellm-secrets
              key: master-key
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: litellm-config
```

### Load Balancing

Use a load balancer to distribute traffic across multiple gateway instances:

- **AWS**: Application Load Balancer
- **GCP**: Cloud Load Balancing
- **Azure**: Application Gateway
- **Kubernetes**: Service with type LoadBalancer

### Scaling

The gateway is stateless and can be scaled horizontally:

1. Deploy multiple instances behind a load balancer
2. Use Redis for shared rate limiting and caching
3. Use PostgreSQL for shared spend tracking

### Security

1. **Use HTTPS**: Terminate TLS at the load balancer or gateway
2. **Use strong API keys**: Use cryptographically secure random keys
3. **Rotate keys regularly**: Implement key rotation policy
4. **Monitor spend**: Set up alerts for unusual spend patterns
5. **Rate limiting**: Configure appropriate rate limits per key

## Performance Tuning

### Connection Pooling

The gateway uses connection pooling for upstream providers. Adjust pool size based on load:

```yaml
# In your application code or environment
POOL_SIZE=100
```

### Caching

Enable Redis caching to reduce latency and provider calls:

```yaml
general_settings:
  redis_url: redis://localhost:6379
  cache_ttl: 300  # Cache for 5 minutes
```

### Timeouts

Adjust timeouts based on your use case:

```yaml
general_settings:
  request_timeout: 600  # 10 minutes for long-running requests
```

## Support

For issues and questions:

1. Check the logs for error messages
2. Review this documentation
3. Check the API documentation
4. Open an issue on GitHub

## License

See the main LiteLLM repository for license information.
