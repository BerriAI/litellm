"""
Performance regression tests for LiteLLM embedding paths (SDK and Router).

Run with: pytest tests/regression_tests/test_embedding_performance.py -v
Update baselines: pytest tests/regression_tests/test_embedding_performance.py -v --update-baseline
"""

import os
import sys
import time
from statistics import median, quantiles

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import Router
from regression_utils import (
    check_regression,
    get_regression_threshold,
    save_baseline,
)

# Configuration
REGRESSION_THRESHOLD = get_regression_threshold()
FAKE_ENDPOINT = os.getenv(
    "FAKE_OPENAI_ENDPOINT", "https://exampleopenaiendpoint-production.up.railway.app"
)
NUM_REQUESTS = int(os.getenv("REGRESSION_NUM_REQUESTS", "250"))


@pytest.mark.asyncio
async def test_sdk_embedding_performance(baselines, update_baseline):
    """Test SDK async embedding latency against fake endpoint."""
    test_name = "sdk_embedding"
    num_requests = NUM_REQUESTS
    
    times = []
    for i in range(num_requests):
        start = time.perf_counter()
        try:
            await litellm.aembedding(
                model="text-embedding-ada-002",
                input=["Hello world"],
                api_base=FAKE_ENDPOINT,
                api_key="fake-key",
                timeout=10,
            )
            times.append((time.perf_counter() - start) * 1000)  # Convert to ms
        except Exception as e:
            pytest.fail(f"Request {i+1}/{num_requests} failed: {e}")
    
    times.sort()
    median_ms = median(times)
    # Calculate P95 using quantiles for accuracy
    percentiles = quantiles(times, n=100)
    p95_ms = percentiles[94]  # 95th percentile is at index 94
    
    if update_baseline:
        save_baseline(test_name, median_ms, p95_ms)
        pytest.skip(f"Baseline updated: median={median_ms:.2f}ms, p95={p95_ms:.2f}ms")
    else:
        check_regression(test_name, median_ms, p95_ms, baselines, REGRESSION_THRESHOLD)


@pytest.mark.asyncio
async def test_router_embedding_performance(baselines, update_baseline):
    """Test Router async embedding latency."""
    test_name = "router_embedding"
    num_requests = NUM_REQUESTS
    
    model_list = [
        {
            "model_name": "text-embedding-ada-002",
            "litellm_params": {
                "model": "text-embedding-ada-002",
                "api_key": "fake-key",
                "api_base": FAKE_ENDPOINT,
            },
        },
    ]
    
    router = Router(model_list=model_list, num_retries=0, timeout=10)
    
    times = []
    for i in range(num_requests):
        start = time.perf_counter()
        try:
            await router.aembedding(
                model="text-embedding-ada-002",
                input=["Hello world"],
            )
            times.append((time.perf_counter() - start) * 1000)
        except Exception as e:
            pytest.fail(f"Request {i+1}/{num_requests} failed: {e}")
    
    times.sort()
    median_ms = median(times)
    # Calculate P95 using quantiles for accuracy
    percentiles = quantiles(times, n=100)
    p95_ms = percentiles[94]  # 95th percentile is at index 94
    
    if update_baseline:
        save_baseline(test_name, median_ms, p95_ms)
        pytest.skip(f"Baseline updated: median={median_ms:.2f}ms, p95={p95_ms:.2f}ms")
    else:
        check_regression(test_name, median_ms, p95_ms, baselines, REGRESSION_THRESHOLD)


if __name__ == "__main__":
    # Run with baseline update
    sys.exit(pytest.main([__file__, "-v", "--update-baseline"]))

