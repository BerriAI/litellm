# LiteLLM Proxy Benchmark: Standard vs fast-litellm

## Overview

This benchmark compares the **standard LiteLLM proxy** against the **[fast-litellm](https://github.com/neul-labs/fast-litellm) accelerated proxy** (v0.1.6), which uses Rust via PyO3 to speed up internal proxy operations.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Mode | `network_mock` (pure proxy overhead, no real API calls) |
| Database | Local PostgreSQL (eliminates network DB latency) |
| Workers | 4 uvicorn workers |
| Requests per level | 2,000 |
| Runs per level | 5 (median taken) |
| Concurrency levels | 10, 50, 100, 200 |
| Python | 3.12 |
| fast-litellm | v0.1.6 |
| LiteLLM | v1.82.2 |

## Results

### Throughput (requests/second)

| Concurrency | Standard | fast-litellm | Speedup |
|:-----------:|:--------:|:------------:|:-------:|
| 10 | 822 rps | 510 rps | 0.62x |
| 50 | 493 rps | 478 rps | 0.97x |
| 100 | 809 rps | 822 rps | 1.02x |
| 200 | 530 rps | 585 rps | 1.10x |

### Mean Latency (ms)

| Concurrency | Standard | fast-litellm | Change |
|:-----------:|:--------:|:------------:|:------:|
| 10 | 12.0 ms | 19.3 ms | +60.8% |
| 50 | 99.9 ms | 101.8 ms | +1.9% |
| 100 | 117.8 ms | 115.3 ms | -2.2% |
| 200 | 313.9 ms | 323.1 ms | +2.9% |

### Tail Latency

| Concurrency | Metric | Standard | fast-litellm | Change |
|:-----------:|:------:|:--------:|:------------:|:------:|
| 100 | P95 | 238.6 ms | 219.1 ms | -8.2% |
| 100 | P99 | 397.7 ms | 248.2 ms | **-37.6%** |
| 200 | P95 | 604.9 ms | 575.3 ms | -4.9% |
| 200 | P99 | 853.0 ms | 874.7 ms | +2.5% |

## Key Findings

1. **No significant overall speedup**: fast-litellm v0.1.6 does not provide a measurable throughput improvement for the LiteLLM proxy in this benchmark. The overall throughput ratio is **0.90x** (fast-litellm / standard).

2. **Low concurrency regression**: At concurrency=10, the standard proxy is **~38% faster** in throughput and has ~60% lower mean latency. This is likely due to PyO3 monkeypatching overhead being proportionally larger when individual request latency is very low.

3. **High concurrency parity**: At concurrency=100-200, performance is essentially equivalent, with fast-litellm showing slight improvements in P99 tail latency at concurrency=100 (-37.6%).

4. **Partial patch application**: Several fast-litellm patches failed to apply:
   - `SimpleRateLimiter` class not found in litellm
   - `SimpleConnectionPool` class not found in litellm
   - `count_tokens_batch` function not found in litellm.utils
   
   These classes/functions may have been renamed or restructured in the current LiteLLM version (v1.82.2), limiting fast-litellm's effectiveness.

5. **Bottleneck is I/O, not CPU**: The proxy's main bottleneck is async I/O (database auth lookups, even with local PostgreSQL), not CPU-bound Python operations. Rust-accelerating CPU-bound operations doesn't help when they're not on the critical path.

## Interpretation

The fast-litellm project targets specific CPU-bound operations (connection pooling, rate limiting, token counting) with Rust replacements. In the context of a full proxy request lifecycle — which includes FastAPI routing, authentication, database lookups, request/response transformation, and async I/O — these CPU-bound operations represent a small fraction of total request time.

For the Rust acceleration to show meaningful improvements, the benchmark would need to:
- Exercise rate limiting under high cardinality (1000+ unique keys)
- Include large token counting workloads (the `network_mock` mode returns small mock responses)
- Use the specific connection pooling patterns that fast-litellm optimizes

## How to Reproduce

```bash
# Install dependencies
poetry install
poetry run pip install fast-litellm aiohttp

# Install and start local PostgreSQL
sudo apt-get install -y postgresql
sudo pg_ctlcluster 16 main start
sudo -u postgres createdb litellm_benchmark
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"

# Run standard proxy benchmark
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/litellm_benchmark" \
  poetry run litellm --config benchmark_config_local.yaml --port 4000 --num_workers 4 &
# Wait for health check...
poetry run python scripts/comprehensive_benchmark.py \
  --url "http://localhost:4000/chat/completions" \
  --label "Standard LiteLLM Proxy" \
  --output benchmark_results/standard_comprehensive.json

# Run fast-litellm proxy benchmark
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/litellm_benchmark" \
  poetry run python scripts/start_fast_proxy.py --config benchmark_config_local.yaml --port 4001 --num_workers 4 &
# Wait for health check...
poetry run python scripts/comprehensive_benchmark.py \
  --url "http://localhost:4001/chat/completions" \
  --label "fast-litellm Accelerated Proxy" \
  --output benchmark_results/fast_comprehensive.json

# Generate comparison
poetry run python scripts/generate_comparison.py
```
