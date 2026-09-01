# LiteLLM Rust Gateway - Benchmark Guide

## Overview

This guide explains how to benchmark the LiteLLM Rust Gateway against the Python proxy or other implementations.

## Prerequisites

### Required

- Rust gateway built and running
- Python proxy running (for comparison)
- LLM provider API key(s)

### Recommended

- **Non-rate-limited API endpoint**: For accurate benchmarking, you need an API endpoint that doesn't enforce rate limits. Options:
  - Paid API keys with high rate limits
  - Local LLM (Ollama, LM Studio, etc.)
  - Self-hosted LLM (vLLM, TGI, etc.)
  - Mock server (for relative performance comparison)

**Note**: Free tier API keys (like OpenRouter free models) often have strict rate limits that will skew benchmark results.

## Quick Start Benchmark

### 1. Start the Rust Gateway

```bash
cd litellm-rust
LITELLM_YAML_CONFIG=config.yaml ./target/release/litellm-ai-gateway
```

### 2. Start the Python Proxy (for comparison)

```bash
cd litellm
python -m litellm --config config.yaml --port 4001
```

### 3. Run the Benchmark

```bash
cd litellm-rust/bench
python compare_benchmark.py
```

## Benchmark Scripts

### compare_benchmark.py

Compares Rust gateway vs Python proxy performance.

**Usage:**
```bash
python compare_benchmark.py
```

**What it does:**
- Sends 20 requests to each gateway
- Measures latency (p50, p95, p99)
- Calculates throughput (RPS)
- Compares results

**Expected output:**
```
============================================================
REAL BENCHMARK: Rust Gateway vs Python Proxy
============================================================

Testing: Rust Gateway
  Success rate: 20/20 (100%)
  Mean latency: 1500.0 ms
  Median (p50): 1200.0 ms
  P95 latency: 2500.0 ms
  Throughput: 0.7 RPS

Testing: Python Proxy
  Success rate: 20/20 (100%)
  Mean latency: 4500.0 ms
  Median (p50): 3800.0 ms
  P95 latency: 8000.0 ms
  Throughput: 0.2 RPS
```

### load_test.py

Sustained load test for a single endpoint.

**Usage:**
```bash
python load_test.py --url http://localhost:4001 --duration 60 --concurrency 10
```

**Parameters:**
- `--url`: Gateway URL
- `--duration`: Test duration in seconds
- `--concurrency`: Number of concurrent requests

### chaos_test.py

Tests gateway behavior under failure conditions.

**Usage:**
```bash
python chaos_test.py
```

**What it does:**
- Tests circuit breaker behavior
- Tests retry logic
- Tests fallback routing

## Benchmarking with Different Providers

### Using OpenAI

```yaml
# config.yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: sk-your-openai-key
```

**Note**: OpenAI has rate limits. Use a paid key with high limits for accurate benchmarking.

### Using Anthropic

```yaml
model_list:
  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: sk-ant-your-key
```

### Using Local LLM (Ollama)

1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama2`
3. Configure gateway:

```yaml
model_list:
  - model_name: llama2
    litellm_params:
      model: ollama/llama2
      api_base: http://localhost:11434
```

**Advantages:**
- No rate limits
- No API costs
- Consistent performance

**Disadvantages:**
- Slower than cloud APIs
- Requires local resources

### Using Self-Hosted LLM (vLLM)

1. Install vLLM: https://vllm.ai/
2. Start vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --port 8000
```

3. Configure gateway:

```yaml
model_list:
  - model_name: llama2
    litellm_params:
      model: openai/llama2
      api_base: http://localhost:8000/v1
```

### Using Mock Server (for relative comparison)

For relative performance comparison (not absolute performance):

```bash
cd litellm-rust/bench
python mock_upstream.py &
```

Then configure gateway to use the mock:

```yaml
model_list:
  - model_name: mock
    litellm_params:
      model: openai/mock
      api_base: http://localhost:11434/v1
      api_key: mock-key
```

**Note**: Mock server provides relative performance comparison but not absolute performance metrics.

## Benchmark Metrics

### Key Metrics

1. **Latency**
   - p50 (median): Typical request latency
   - p95: 95th percentile latency
   - p99: 99th percentile latency
   - Mean: Average latency

2. **Throughput**
   - RPS (Requests Per Second): Number of requests handled per second
   - TPS (Tokens Per Second): Number of tokens processed per second

3. **Success Rate**
   - Percentage of successful requests
   - Should be >99% for production

4. **Error Rate**
   - Percentage of failed requests
   - Should be <1% for production

### Interpreting Results

**Rust Gateway Advantages:**
- Lower latency (typically 2-5x faster than Python)
- Higher throughput (typically 3-10x higher RPS)
- Lower resource usage (CPU, memory)
- Better concurrency handling

**Expected Results:**
- Rust gateway: 1000-5000ms latency, 0.5-2.0 RPS (with real LLM)
- Python proxy: 3000-10000ms latency, 0.1-0.5 RPS (with real LLM)

**Note**: Actual numbers depend on:
- LLM provider latency
- Network latency
- Model size
- Request complexity
- Hardware resources

## Production Benchmarking

### Environment Setup

For production benchmarking:

1. **Use dedicated hardware**
   - Separate machines for gateway and LLM provider
   - High-speed network connection
   - Sufficient CPU/memory

2. **Use production-like configuration**
   - Same config as production
   - Same LLM provider
   - Same rate limits

3. **Run for extended duration**
   - Minimum 5 minutes
   - Recommended 15-30 minutes
   - Test different concurrency levels

### Benchmark Script for Production

```bash
#!/bin/bash
# production_benchmark.sh

URL=${1:-http://localhost:4001}
DURATION=${2:-300}  # 5 minutes
CONCURRENCY_LEVELS="1 5 10 25 50 100"

echo "Benchmarking $URL for ${DURATION}s"
echo "Concurrency levels: $CONCURRENCY_LEVELS"
echo ""

for CONCURRENCY in $CONCURRENCY_LEVELS; do
  echo "=== Concurrency: $CONCURRENCY ==="
  python load_test.py --url $URL --duration $DURATION --concurrency $CONCURRENCY
  echo ""
done
```

**Usage:**
```bash
chmod +x production_benchmark.sh
./production_benchmark.sh http://localhost:4001 300
```

## Monitoring During Benchmark

### System Metrics

Monitor system resources during benchmark:

```bash
# CPU and memory
top -d 5

# Network
iftop -i eth0

# Disk I/O
iotop
```

### Gateway Metrics

Monitor gateway metrics:

```bash
# Prometheus metrics
curl http://localhost:4001/metrics

# Key metrics to watch:
# - litellm_requests_total
# - litellm_request_duration_seconds
# - litellm_tokens_total
```

### LLM Provider Metrics

Monitor LLM provider:
- Request rate
- Latency
- Error rate
- Token usage

## Troubleshooting

### Low Throughput

**Possible causes:**
- LLM provider rate limiting
- Network latency
- Insufficient resources
- Configuration issues

**Solutions:**
- Use non-rate-limited API endpoint
- Check network latency
- Increase resources
- Review configuration

### High Latency

**Possible causes:**
- LLM provider latency
- Network latency
- Large model
- Complex requests

**Solutions:**
- Use faster LLM provider
- Optimize network
- Use smaller model
- Simplify requests

### High Error Rate

**Possible causes:**
- LLM provider errors
- Network issues
- Configuration errors
- Rate limiting

**Solutions:**
- Check LLM provider status
- Check network connectivity
- Review configuration
- Increase rate limits

## Benchmark Results Template

Use this template to document benchmark results:

```markdown
## Benchmark Results

**Date:** YYYY-MM-DD
**Gateway Version:** vX.Y.Z
**Hardware:** [CPU, Memory, Network]
**LLM Provider:** [Provider, Model]
**Configuration:** [Key settings]

### Results

| Concurrency | Latency (p50) | Latency (p95) | Throughput (RPS) | Success Rate |
|-------------|---------------|---------------|------------------|--------------|
| 1           | XXXX ms       | XXXX ms       | X.XX RPS         | XX%          |
| 5           | XXXX ms       | XXXX ms       | X.XX RPS         | XX%          |
| 10          | XXXX ms       | XXXX ms       | X.XX RPS         | XX%          |
| 25          | XXXX ms       | XXXX ms       | X.XX RPS         | XX%          |
| 50          | XXXX ms       | XXXX ms       | X.XX RPS         | XX%          |
| 100         | XXXX ms       | XXXX ms       | X.XX RPS         | XX%          |

### Analysis

[Analysis of results, comparisons, observations]

### Recommendations

[Recommendations based on results]
```

## Additional Resources

- [User Guide](USER_GUIDE.md) - Configuration and usage
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment
- [API Documentation](API_DOCUMENTATION.md) - API reference
- [Project Summary](PROJECT_SUMMARY.md) - Project overview

## Support

For issues with benchmarking:

1. Check that you're using a non-rate-limited API endpoint
2. Verify network connectivity
3. Check system resources
4. Review gateway logs
5. Check LLM provider status

For bugs or issues, open an issue on GitHub.
