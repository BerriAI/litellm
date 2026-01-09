"""
Memray Memory Leak Detection Tests

Simple tests using pytest-memray to detect memory leaks in LiteLLM operations.
These tests will fail if memory leaks are detected, helping catch OOM issues before production.

NOTE: Memray requires Linux/WSL. To run these tests:
1. Install in WSL/Linux: pip install pytest-memray
2. Run: pytest tests/load_tests/test_memray_leak_detection.py --memray -v

For Windows without WSL, these tests will be skipped automatically.
"""

import os
import sys

# Add parent directory to path to import litellm (same pattern as other tests in this directory)
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(filepath, "../..")))

import asyncio
import gc
import random

import pytest
import psutil
from litellm.router import Router

# Check if pytest-memray is available
try:
    import pytest_memray
    MEMRAY_AVAILABLE = True
except ImportError:
    MEMRAY_AVAILABLE = False

# Test Configuration Constants
TEST_API_BASE = "https://exampleopenaiendpoint-production-0ee2.up.railway.app/"
TEST_API_KEY = "sk-1234"
TEST_MODEL_NAME = "gpt-3.5-turbo"

# Memory Test Configuration Constants
TOTAL_BATCHES = 500
REQUESTS_PER_BATCH = 5
MEDIUM_CHECKPOINT_BATCH = 250  # Measure memory after this many batches

# Leak Detection Thresholds
MAX_TOTAL_GROWTH_MB = 200.0  # Fail if total growth exceeds this (MB) - adjust to control sensitivity

# Request Pattern Constants
CONCURRENT_BURST_INTERVAL = 10  # Every N batches, do concurrent requests
CONCURRENT_BURST_MIN = 5
CONCURRENT_BURST_MAX = 10

# Message Size Configuration
SHORT_MESSAGE_MULTIPLIER = 1
MEDIUM_MESSAGE_MULTIPLIER = 10  # ~200 chars
LONG_MESSAGE_MULTIPLIER = 50  # ~1000 chars

# Request Parameter Variation
TEMPERATURE_MODULO = 5  # Add temperature every N requests
TEMPERATURE_MIN = 0.5
TEMPERATURE_MAX = 1.5
MAX_TOKENS_MODULO = 7  # Add max_tokens every N requests
MAX_TOKENS_MIN = 50
MAX_TOKENS_MAX = 200

# Timing Constants (seconds)
INITIAL_STABILIZATION_DELAY = 0.1
GC_STABILIZATION_DELAY = 0.05

# Create router instance (same as proxy server uses)
test_router = Router(
    model_list=[
        {
            "model_name": TEST_MODEL_NAME,
            "litellm_params": {
                "model": f"openai/{TEST_MODEL_NAME}",
                "api_base": TEST_API_BASE,
                "api_key": TEST_API_KEY,
            },
        },
    ]
)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in megabytes.
    
    Returns:
        Current RSS (Resident Set Size) memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def generate_message_content(request_id: int) -> str:
    """Generate varied message content to simulate production patterns.
    
    Args:
        request_id: Unique identifier for the request.
        
    Returns:
        Message content string with varying lengths.
    """
    if request_id % 3 == 0:
        return f"Short request {request_id}"
    elif request_id % 3 == 1:
        return f"Medium request {request_id}. " * MEDIUM_MESSAGE_MULTIPLIER
    else:
        return f"Long request {request_id}. " * LONG_MESSAGE_MULTIPLIER


async def make_varied_request(router: Router, request_id: int) -> object:
    """Make a request with varying patterns to simulate production usage.
    
    Args:
        router: LiteLLM Router instance to use for requests.
        request_id: Unique identifier for the request.
        
    Returns:
        Response object from the router.
    """
    content = generate_message_content(request_id)
    
    kwargs = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
    }
    
    # Occasionally add temperature parameter
    if request_id % TEMPERATURE_MODULO == 0:
        kwargs["temperature"] = random.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX)
    
    # Occasionally add max_tokens parameter
    if request_id % MAX_TOKENS_MODULO == 0:
        kwargs["max_tokens"] = random.randint(MAX_TOKENS_MIN, MAX_TOKENS_MAX)
    
    return await router.acompletion(**kwargs)


async def execute_sequential_requests(
    router: Router, batch: int, requests_per_batch: int
) -> None:
    """Execute sequential requests for a batch.
    
    Does not return response objects to prevent test code from accumulating memory.
    Only asserts that responses are valid, then immediately discards them.
    
    Args:
        router: LiteLLM Router instance.
        batch: Current batch number.
        requests_per_batch: Number of requests to make in this batch.
    """
    for i in range(requests_per_batch):
        request_id = batch * requests_per_batch + i
        response = await make_varied_request(router, request_id)
        assert response is not None, f"Request {request_id} returned None"
        # Explicitly delete response to ensure cleanup
        del response
        # Aggressive GC after each request
        gc.collect()


async def execute_concurrent_requests(
    router: Router, batch: int, requests_per_batch: int, concurrent_count: int
) -> None:
    """Execute concurrent requests for a batch (simulates traffic spikes).
    
    Does not return response objects to prevent test code from accumulating memory.
    Only asserts that responses are valid, then immediately discards them.
    
    Args:
        router: LiteLLM Router instance.
        batch: Current batch number.
        requests_per_batch: Base number of requests per batch.
        concurrent_count: Number of concurrent requests to make.
    """
    tasks = [
        make_varied_request(router, batch * requests_per_batch + i)
        for i in range(concurrent_count)
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=False)
    assert all(r is not None for r in responses), "Some concurrent requests returned None"
    # Explicitly delete responses to ensure cleanup
    del responses
    del tasks
    # Aggressive GC after concurrent batch
    gc.collect()


async def measure_memory_after_gc() -> float:
    """Force garbage collection and measure memory usage.
    
    Returns:
        Memory usage in MB after GC.
    """
    gc.collect()
    await asyncio.sleep(GC_STABILIZATION_DELAY)
    return get_memory_usage_mb()


@pytest.mark.skipif(
    sys.platform == "win32" or not MEMRAY_AVAILABLE,
    reason="Memray requires Linux/WSL and pytest-memray. Install with: pip install pytest-memray (Linux/WSL only)"
)
@pytest.mark.asyncio
@pytest.mark.limit_leaks("30 MB")
async def test_extended_memory_growth():
    """
    Simple memory leak test with three measurements:
    1. Baseline: before any requests
    2. Medium: after MEDIUM_CHECKPOINT_BATCH batches
    3. Final: after all batches
    
    Uses aggressive GC and minimal test state (only a simple int counter).
    Fails if total memory growth exceeds MAX_TOTAL_GROWTH_MB.
    
    Features:
    - Varying message sizes (short and long prompts)
    - Mix of sequential and concurrent requests
    - Different request parameters (temperature, max_tokens)
    - Aggressive garbage collection after each batch
    """
    # Measure baseline memory (before any requests)
    baseline_memory = await measure_memory_after_gc()
    print(f"\n[Memory Test] Baseline memory: {baseline_memory:.2f} MB")
    
    # Only keep a simple int counter - no lists or other data structures
    batch_counter = 0
    
    # Execute request batches
    for batch_counter in range(TOTAL_BATCHES):
        # Execute concurrent burst requests periodically to simulate traffic spikes
        if batch_counter % CONCURRENT_BURST_INTERVAL == 0 and batch_counter > 0:
            concurrent_count = random.randint(CONCURRENT_BURST_MIN, CONCURRENT_BURST_MAX)
            await execute_concurrent_requests(
                test_router, batch_counter, REQUESTS_PER_BATCH, concurrent_count
            )
            # Explicitly delete local variable to ensure cleanup
            del concurrent_count
        else:
            # Execute sequential requests (normal traffic pattern)
            await execute_sequential_requests(test_router, batch_counter, REQUESTS_PER_BATCH)
        
        # Aggressive GC after each batch to prevent test data accumulation
        gc.collect()
        
        # Measure medium checkpoint memory
        if batch_counter == MEDIUM_CHECKPOINT_BATCH:
            medium_memory = await measure_memory_after_gc()
            medium_growth = medium_memory - baseline_memory
            print(
                f"[Memory Test] Medium checkpoint (batch {MEDIUM_CHECKPOINT_BATCH}): "
                f"{medium_memory:.2f} MB (growth: {medium_growth:+.2f} MB)"
            )
            # Check if growth already exceeds threshold
            if medium_growth > MAX_TOTAL_GROWTH_MB:
                pytest.fail(
                    f"Memory leak detected at medium checkpoint: "
                    f"Growth of {medium_growth:.2f} MB exceeds threshold of {MAX_TOTAL_GROWTH_MB} MB. "
                    f"Baseline: {baseline_memory:.2f} MB, Medium: {medium_memory:.2f} MB"
                )
    
    # Final memory measurement
    final_memory = await measure_memory_after_gc()
    final_growth = final_memory - baseline_memory
    
    print(
        f"\n[Memory Test] Final memory: {final_memory:.2f} MB "
        f"(total growth: {final_growth:+.2f} MB from baseline {baseline_memory:.2f} MB)"
    )
    print(f"[Memory Test] Processed {TOTAL_BATCHES} batches (~{TOTAL_BATCHES * REQUESTS_PER_BATCH} requests)")
    
    # Check if final growth exceeds threshold
    if final_growth > MAX_TOTAL_GROWTH_MB:
        pytest.fail(
            f"Memory leak detected: "
            f"Total growth of {final_growth:.2f} MB exceeds threshold of {MAX_TOTAL_GROWTH_MB} MB. "
            f"Baseline: {baseline_memory:.2f} MB, Final: {final_memory:.2f} MB"
        )
    
    print("[Memory Test] Test completed successfully - no memory leak detected.")