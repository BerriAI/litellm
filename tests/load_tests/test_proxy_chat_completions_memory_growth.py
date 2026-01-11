"""
Memory Leak Detection Tests - Proxy Server Linear Memory Growth

Tests that check for linear/progressive memory growth in the proxy server by running 
different numbers of requests (1k, 2k, 4k, 10k, 30k) with the same memory limit.

These tests start an actual proxy server (not just the Router) and make HTTP requests
to it, simulating real-world usage more accurately.

IMPORTANT: These tests should be run INDIVIDUALLY, not all together. Running them
together causes memory baseline drift between tests, making it difficult to detect
linear growth accurately. Each test should be run in isolation:

    pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_1k -v
    pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_2k -v
    # etc.

NOTE: Not recommended for accurate results:
pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py -v
"""

import asyncio
import gc
import os
import sys
import tempfile
import time

import httpx
import psutil
import pytest
import yaml

# Add parent directory to path to import litellm (same pattern as other tests)
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(filepath, "../..")))

from tests.load_tests.memory_leak_utils import (
    limit_memory,  # noqa: F401  # pytest fixture used via dependency injection
    mock_server,  # noqa: F401  # pytest fixture used via dependency injection
)

# Memory limit for all linear memory growth tests
MEMORY_LIMIT = "45 MB"

# Test Configuration
TEST_API_KEY = "sk-1234"
TEST_MODEL_NAME = "db-openai-endpoint"


@pytest.fixture
async def proxy_server(mock_server, monkeypatch):
    """
    Fixture to start a proxy server with config for each test.
    
    Creates a temporary config file, starts the proxy server,
    and yields the base URL for making requests.
    """
    from litellm.proxy.proxy_server import app, cleanup_router_config_variables, initialize
    
    # Clean up any previous router config
    cleanup_router_config_variables()
    
    # Create config that points to mock server
    config = {
        "model_list": [
            {
                "model_name": TEST_MODEL_NAME,
                "litellm_params": {
                    "model": "openai/*",
                    "api_base": mock_server,
                },
            }
        ],
        "general_settings": {
            "master_key": TEST_API_KEY,
        },
        "router_settings": {
            "disable_cooldowns": True,  # Disable cooldowns for testing
            "allowed_fails": 1000,  # Allow many failures before cooldown
        },
    }
    
    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Initialize proxy server
        print("[Proxy Setup] Initializing proxy server...")
        start_init = time.time()
        await initialize(config=config_path, debug=False)
        print(f"[Proxy Setup] Proxy initialized in {time.time() - start_init:.2f}s")
        
        # Create an ASGI test client using httpx with optimized settings for high concurrency
        # Note: We use httpx.AsyncClient with app transport for in-process testing
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            limits=httpx.Limits(
                max_keepalive_connections=1000,  # Allow many concurrent connections
                max_connections=1000,
                keepalive_expiry=30.0,
            ),
        ) as client:
            # Warmup request to trigger any lazy loading
            print("[Proxy Setup] Sending warmup request...")
            start_warmup = time.time()
            try:
                warmup_response = await client.post(
                    "/chat/completions",
                    json={
                        "model": TEST_MODEL_NAME,
                        "messages": [{"role": "user", "content": "warmup"}],
                    },
                    headers={"Authorization": f"Bearer {TEST_API_KEY}"},
                    timeout=10.0,
                )
                print(f"[Proxy Setup] Warmup completed in {time.time() - start_warmup:.2f}s (status: {warmup_response.status_code})")
            except Exception as e:
                print(f"[Proxy Setup] Warmup failed: {e}")
            
            print("[Proxy Setup] Server ready for load testing!")
            yield client
        
        # Cleanup after test
        cleanup_router_config_variables()
    finally:
        # Remove temporary config file
        try:
            os.unlink(config_path)
        except Exception:
            pass


async def run_proxy_memory_baseline_test(
    num_requests: int, 
    proxy_client: httpx.AsyncClient, 
    limit_memory
):
    """
    Helper function to run memory baseline test against proxy server.
    
    Makes HTTP requests to the proxy server (not Router directly) to test
    memory usage in a more realistic scenario.
    
    Memory Management Strategy:
    - Explicit cleanup of all variables after each batch
    - Multiple GC passes at the end to ensure test artifacts are cleaned
    - Memory tracking throughout to detect any gradual accumulation
    - Ensures that only proxy server memory (not test artifacts) is measured
    
    Args:
        num_requests: Number of requests to make.
        proxy_client: httpx AsyncClient configured to talk to proxy server.
        limit_memory: Pytest fixture for memory tracking.
    
    Example:
        @pytest.mark.asyncio
        @pytest.mark.limit_leaks("40 MB")
        async def test_memory(proxy_server, limit_memory):
            await run_proxy_memory_baseline_test(1000, proxy_server, limit_memory)
    """
    # Fixture is used automatically by pytest - reference it to suppress linter warning
    _ = limit_memory
    
    # Track memory throughout test to ensure no accumulation
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Memory] Test start: {start_memory:.2f} MB")
    
    # Make requests concurrently in large batches for speed
    # Large batch size = faster tests (temporary memory spike is accounted for in limit)
    BATCH_SIZE = 100
    
    # Pre-build request data to avoid repeated dict creation
    request_headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    
    batch_num = 0
    total_batches = (num_requests + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_start in range(0, num_requests, BATCH_SIZE):
        batch_num += 1
        batch_end = min(batch_start + BATCH_SIZE, num_requests)
        
        # Create tasks directly without intermediate function (faster)
        # Use minimal timeout and skip response validation for speed
        tasks = [
            proxy_client.post(
                "/chat/completions",
                json={
                    "model": TEST_MODEL_NAME,
                    "messages": [{"role": "user", "content": f"Test request {i}"}],
                },
                headers=request_headers,
                timeout=30.0,  # Longer timeout to avoid failures under load
            )
            for i in range(batch_start, batch_end)
        ]
        
        print(f"[Batch {batch_num}/{total_batches}] Executing batch {batch_start}-{batch_end}...")
        batch_start_time = time.time()
        # Execute batch concurrently (return_exceptions=True to not fail on individual errors)
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        batch_duration = time.time() - batch_start_time
        
        # Clean up batch immediately to avoid memory accumulation
        # Explicitly clear each response to break any circular references
        if responses:
            for resp in responses:
                del resp
        del responses
        del tasks
        
        # Periodic memory check to detect gradual accumulation
        if batch_num % 10 == 0 or batch_num == total_batches:
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"[Batch {batch_num}/{total_batches}] Completed in {batch_duration:.2f}s ({BATCH_SIZE/batch_duration:.1f} req/s) | Memory: {current_memory:.2f} MB (+{current_memory - start_memory:.2f} MB)")
        else:
            print(f"[Batch {batch_num}/{total_batches}] Completed in {batch_duration:.2f}s ({BATCH_SIZE/batch_duration:.1f} req/s)")
    
    # Explicit cleanup of all test artifacts
    del request_headers
    
    # Check memory before cleanup
    memory_before_cleanup = process.memory_info().rss / 1024 / 1024
    print(f"\n[Memory] Before cleanup: {memory_before_cleanup:.2f} MB (+{memory_before_cleanup - start_memory:.2f} MB)")
    
    # EXTREMELY AGGRESSIVE CLEANUP to ensure test memory doesn't pollute measurements
    print("[Cleanup] Starting EXTRA AGGRESSIVE garbage collection...")
    
    # Step 1: Force immediate cleanup of all generations
    collected_total = 0
    for gen in range(3):
        collected = gc.collect(gen)
        collected_total += collected
        if collected > 0:
            print(f"[Cleanup] Generation {gen}: collected {collected} objects")
    
    # Step 2: Multiple full GC passes to catch circular references
    print("[Cleanup] Running full GC passes...")
    for i in range(5):  # Increased from 3 to 5 passes
        collected = gc.collect()
        collected_total += collected
        if collected > 0:
            print(f"[Cleanup] Full GC pass {i+1}: collected {collected} objects")
        else:
            print(f"[Cleanup] Full GC pass {i+1}: no objects to collect (clean!)")
    
    # Step 3: Check for uncollectable objects (memory leaks!)
    uncollectable = len(gc.garbage)
    if uncollectable > 0:
        print(f"[Cleanup] ⚠️  WARNING: {uncollectable} uncollectable objects found (potential leak!)")
        print(f"[Cleanup] Uncollectable types: {set(type(obj).__name__ for obj in gc.garbage[:10])}")
    else:
        print(f"[Cleanup] ✅ No uncollectable objects (no circular reference leaks)")
    
    # Step 4: Wait longer for OS to release memory
    print("[Cleanup] Waiting for OS to release memory...")
    time.sleep(0.5)  # Increased from 0.1s to 0.5s
    
    # Step 5: Final GC to catch anything released during wait
    final_gc = gc.collect()
    collected_total += final_gc
    if final_gc > 0:
        print(f"[Cleanup] Final GC: collected {final_gc} additional objects")
    
    # Final memory check
    memory_after_cleanup = process.memory_info().rss / 1024 / 1024
    print(f"\n[Memory] ✅ After cleanup: {memory_after_cleanup:.2f} MB (+{memory_after_cleanup - start_memory:.2f} MB from start)")
    print(f"[Memory] 🗑️  Cleanup freed: {memory_before_cleanup - memory_after_cleanup:.2f} MB")
    print(f"[Memory] 📦 Total objects collected: {collected_total:,}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"[Proxy Memory Test] ✅ Completed {num_requests:,} requests")
    print(f"[Proxy Memory Test] 📊 Actual proxy memory usage: {memory_after_cleanup - start_memory:.2f} MB")
    print(f"[Proxy Memory Test] 🧹 Test artifacts cleaned: {memory_before_cleanup - memory_after_cleanup:.2f} MB")
    print(f"{'='*80}\n")


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_1k(proxy_server, limit_memory):
    """
    Memory baseline test with 1,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If this passes but higher request count tests fail, indicates progressive memory leak.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_1k -v
    """
    await run_proxy_memory_baseline_test(1000, proxy_server, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_2k(proxy_server, limit_memory):
    """
    Memory baseline test with 2,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If this passes but test_proxy_memory_baseline_4k fails, indicates progressive memory leak.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_2k -v
    """
    await run_proxy_memory_baseline_test(2000, proxy_server, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_4k(proxy_server, limit_memory):
    """
    Memory baseline test with 4,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_proxy_memory_baseline_1k and test_proxy_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_4k -v
    """
    await run_proxy_memory_baseline_test(4000, proxy_server, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_10k(proxy_server, limit_memory):
    """
    Memory baseline test with 10,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_proxy_memory_baseline_1k and test_proxy_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_10k -v
    """
    await run_proxy_memory_baseline_test(10000, proxy_server, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_15k(proxy_server, limit_memory):
    """
    Memory baseline test with 15,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_proxy_memory_baseline_1k and test_proxy_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_15k -v
    """
    await run_proxy_memory_baseline_test(15000, proxy_server, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_20k(proxy_server, limit_memory):
    """
    Memory baseline test with 20,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_proxy_memory_baseline_1k and test_proxy_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_20k -v
    """
    await run_proxy_memory_baseline_test(20000, proxy_server, limit_memory)



@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_25k(proxy_server, limit_memory):
    """
    Memory baseline test with 25,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_proxy_memory_baseline_1k and test_proxy_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_25k -v
    """
    await run_proxy_memory_baseline_test(25000, proxy_server, limit_memory)

@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_30k(proxy_server, limit_memory):
    """
    Memory baseline test with 30,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_proxy_memory_baseline_1k and test_proxy_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_30k -v
    """
    await run_proxy_memory_baseline_test(30000, proxy_server, limit_memory)



# We're only supposed to make it here if we're in a good place, meaning the memory limit needs to be strict enough to catch any possible OOMs from the previous tests.
@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_50k(proxy_server, limit_memory):
    """
    Memory baseline test with 50,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_proxy_memory_baseline_1k and test_proxy_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_50k -v
    """
    await run_proxy_memory_baseline_test(50000, proxy_server, limit_memory)

@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_500k(proxy_server, limit_memory):
    """
    Memory baseline test with 500,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_proxy_memory_baseline_1k and test_proxy_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_500k -v
    """
    await run_proxy_memory_baseline_test(500000, proxy_server, limit_memory)


@pytest.mark.asyncio
@pytest.mark.limit_leaks(MEMORY_LIMIT)
@pytest.mark.no_parallel  # Must run sequentially - measures process memory
async def test_proxy_memory_baseline_1m(proxy_server, limit_memory):
    """
    Memory baseline test with 1,000,000 requests to the proxy server.
    Uses @pytest.mark.limit_leaks("40 MB") to enforce memory limit.
    If test_proxy_memory_baseline_1k and test_proxy_memory_baseline_2k pass but this fails,
    it's a clear sign of sequential/progressive memory growth.
    
    NOTE: This test should be run INDIVIDUALLY, not with other tests in this file.
    Running multiple tests together causes memory baseline drift, making it difficult
    to accurately detect linear memory growth. Run with:
        pytest tests/load_tests/test_proxy_chat_completions_memory_growth.py::test_proxy_memory_baseline_1m -v
    """
    await run_proxy_memory_baseline_test(1000000, proxy_server, limit_memory)
