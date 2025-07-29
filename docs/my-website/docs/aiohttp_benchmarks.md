# LiteLLM v1.71.1 Benchmarks

## Overview

This document presents performance benchmarks comparing LiteLLM's v1.71.1 to prior litellm versions.

**Related PR:** [#11097](https://github.com/BerriAI/litellm/pull/11097)

## Testing Methodology

The load testing was conducted using the following parameters:
- **Request Rate:** 200 RPS (Requests Per Second)
- **User Ramp Up:** 200 concurrent users
- **Transport Comparison:** httpx (existing) vs aiohttp (new implementation)
- **Number of pods/instance of litellm:** 1
- **Machine Specs:** 2 vCPUs, 4GB RAM
- **LiteLLM Settings:**
    - Tested against a [fake openai endpoint](https://exampleopenaiendpoint-production.up.railway.app/)
    - Set `USE_AIOHTTP_TRANSPORT="True"` in the environment variables. This feature flag enables the aiohttp transport.


## Benchmark Results

| Metric | httpx (Existing) | aiohttp (LiteLLM v1.71.1) | Improvement | Calculation |
|--------|------------------|-------------------|-------------|-------------|
| **RPS** | 50.2 | 224 | **+346%** ✅ | (224 - 50.2) / 50.2 × 100 = 346% |
| **Median Latency** | 2,500ms | 74ms | **-97%** ✅ | (74 - 2500) / 2500 × 100 = -97% |
| **95th Percentile** | 5,600ms | 250ms | **-96%** ✅ | (250 - 5600) / 5600 × 100 = -96% |
| **99th Percentile** | 6,200ms | 330ms | **-95%** ✅ | (330 - 6200) / 6200 × 100 = -95% |

## Key Improvements

- **4.5x increase** in requests per second (from 50.2 to 224 RPS)
- **97% reduction** in median response time (from 2.5 seconds to 74ms)
- **96% reduction** in 95th percentile latency (from 5.6 seconds to 250ms)
- **95% reduction** in 99th percentile latency (from 6.2 seconds to 330ms)


