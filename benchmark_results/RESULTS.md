# Benchmark Results: fast-litellm vs Standard LiteLLM Proxy

**Date**: 2026-03-16  
**Environment**: 4 CPUs, 15 GB RAM, Linux (cloud VM)  
**LiteLLM version**: 1.82.2 (dev)  
**fast-litellm version**: 0.1.6  
**Python**: 3.12  

## Methodology

All scenarios use `network_mock: true` which intercepts at the httpx transport layer, returning
canned OpenAI ChatCompletion responses. This isolates **pure proxy overhead** — routing, auth,
serialization, token counting, spend logging, and DB writes — from actual LLM API latency.

**Load parameters**: 2,000 requests per run, 100 concurrent connections, 3 runs per scenario,
50 warmup requests before each timed run.

> **Note**: The proxy connects to a remote Neon PostgreSQL database for Prisma operations
> (spend tracking, key auth, etc.). This remote DB adds ~1s per-request overhead that dominates
> the results. The relative comparison between scenarios is still valid since all face the same
> DB overhead.

## Results Summary

| Metric | Scenario 1: Baseline | Scenario 2: Standard | Scenario 2b: Standard | Scenario 3: fast-litellm |
|--------|---------------------|----------------------|----------------------|--------------------------|
| **Server** | uvicorn | gunicorn (CLI) | gunicorn (direct) | gunicorn + fast-litellm |
| **Workers** | 1 | 4 | 4 | 4 |
| **Throughput** | **19 req/s** | **76 req/s** | **78 req/s** | **79 req/s** |
| **Mean latency** | 5,090 ms | 1,286 ms | 1,245 ms | **1,239 ms** |
| **P50 latency** | 4,893 ms | 1,177 ms | 1,149 ms | **1,060 ms** |
| **P95 latency** | 5,592 ms | 2,182 ms | 1,970 ms | **2,176 ms** |
| **P99 latency** | 23,132 ms | 3,019 ms | 2,296 ms | **2,624 ms** |
| **Failures** | 0 | 0 | 0 | 0 |
| **Latency CoV** | 1.3% | 3.3% | 4.0% | **0.9%** |
| **Throughput CoV** | 1.4% | 3.8% | 4.0% | **0.2%** |

## Head-to-Head: Standard vs fast-litellm (4 workers, same gunicorn setup)

Comparing Scenario 2b (standard) vs Scenario 3 (fast-litellm) — identical gunicorn configuration:

| Metric | Standard (2b) | fast-litellm (3) | Difference |
|--------|---------------|------------------|------------|
| **Throughput** | 78 req/s | 79 req/s | **+1.3% faster** |
| **Mean latency** | 1,245 ms | 1,239 ms | **-0.5% (6 ms faster)** |
| **P50 latency** | 1,149 ms | 1,060 ms | **-7.7% (89 ms faster)** |
| **P95 latency** | 1,970 ms | 2,176 ms | +10.5% (206 ms slower) |
| **P99 latency** | 2,296 ms | 2,624 ms | +14.3% (328 ms slower) |
| **Run-to-run variance** | 4.0% CoV | 0.9% CoV | **Much more consistent** |

## Analysis

### Key Findings

1. **Single → Multi-worker is the big win**: Going from 1 to 4 workers gives a **4x throughput improvement** (19 → 78 req/s). This is the most impactful optimization.

2. **fast-litellm provides marginal improvement in median latency**: P50 is ~8% better with fast-litellm (1,060 ms vs 1,149 ms), meaning the typical request is measurably faster.

3. **fast-litellm is significantly more consistent**: The run-to-run variance (CoV) drops from 4.0% to 0.9% for latency and from 4.0% to 0.2% for throughput. This means more predictable performance in production.

4. **Tail latencies are mixed**: P95 and P99 are slightly higher with fast-litellm. This could be due to the overhead of the Rust↔Python FFI bridge for the patched components, or normal variance in the tail.

5. **DB overhead dominates**: The ~1,000+ ms per-request latency is primarily from the remote Neon PostgreSQL database (spend tracking, auth lookups). The actual proxy routing/serialization overhead is a small fraction of this. In a production setup with a local PostgreSQL instance, the latency would be much lower, and fast-litellm's improvements would be proportionally more visible.

### Where fast-litellm Shines

According to the [fast-litellm benchmarks](https://github.com/neul-labs/fast-litellm), the Rust acceleration provides:
- **3.2x faster** connection pooling
- **1.6x faster** rate limiting  
- **1.5-1.7x faster** token counting for large texts
- **42x more memory efficient** for high-cardinality rate limiting

These benefits are most apparent in:
- High-throughput scenarios (1000+ QPS)
- Large token counting workloads
- High-cardinality rate limiting (many unique API keys)
- Memory-constrained environments

In our benchmark, the remote DB bottleneck masks most of these gains. With a local DB, the differences would be more pronounced.

### Limitations of This Benchmark

- **Remote database**: The Neon PostgreSQL connection adds ~1s overhead per request, overwhelming the proxy's internal processing time.
- **Limited Rust patching**: fast-litellm reports `SimpleRateLimiter` and `SimpleConnectionPool` classes not found in the current LiteLLM version, meaning only token counting and routing acceleration are active.
- **Small text payloads**: The benchmark uses minimal message sizes; fast-litellm's token counting advantage scales with text length.
- **Single machine**: Both proxy and benchmark client run on the same 4-CPU VM, causing resource contention.

## Raw Results

See individual scenario output files in this directory:
- `scenario1_baseline.txt` — uvicorn, 1 worker
- `scenario2_standard_multiworker.txt` — gunicorn 4 workers (via CLI)
- `scenario2b_standard_gunicorn_direct.txt` — gunicorn 4 workers (direct, for fair comparison)
- `scenario3_fast_litellm.txt` — gunicorn 4 workers + fast-litellm
