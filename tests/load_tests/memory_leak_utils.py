"""
Memory Leak Testing Utilities

This module provides reusable utilities, fixtures, and helpers for memory leak
and OOM (Out of Memory) detection tests. It includes:
- Mock server setup for local testing
- Memory tracking fixtures
- Router fixtures configured for testing
- Helper functions for running memory baseline tests

Usage:
    from tests.load_tests.memory_leak_utils import (
        mock_server,
        limit_memory,
        test_router,
        run_memory_baseline_test,
    )
"""

import gc
import os
import socket
import sys
import time
from threading import Thread

# Add parent directory to path to import litellm (same pattern as other tests)
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(filepath, "../..")))

import httpx
import pytest
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from litellm.router import Router

# Test Configuration Constants
TEST_API_KEY = "sk-1234"
TEST_MODEL_NAME = "gpt-3.5-turbo"

# Timing Constants (seconds)
GC_STABILIZATION_DELAY = 0.05


# Mock OpenAI-compatible server
def create_mock_server():
    """Create a simple FastAPI mock server that mimics OpenAI API responses."""
    app = FastAPI()
    
    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        """Mock OpenAI chat completions endpoint."""
        request_data = await request.json()
        # Return a simple mock response
        return JSONResponse({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", TEST_MODEL_NAME),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Mock response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        })
    
    # Catch-all route to see what URLs are being requested
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catch_all(request: Request, path: str):
        """Catch-all route to debug what URLs are being requested."""
        print(f"[Mock Server] Received request: {request.method} {request.url.path}")
        # For non-chat-completions, return 404
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    
    return app


def run_server(app, port):
    """Run uvicorn server in a thread."""
    import uvicorn
    # Use uvicorn.run which blocks - this is fine in a daemon thread
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error", access_log=False)


@pytest.fixture(scope="session")
def mock_server():
    """Start a mock server in a separate thread for the test session.
    
    Yields the server URL (with trailing slash) for use in router configuration.
    
    Dynamically finds an available port to support running multiple tests in parallel.
    """
    app = create_mock_server()
    
    # Try to find an available port dynamically
    # Start with preferred ports, then try a range
    port_candidates = [18888, 18889] + list(range(18890, 18920))
    port = None
    
    for candidate_port in port_candidates:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", candidate_port))
            sock.close()
            port = candidate_port
            print(f"[Mock Server] Found available port: {port}")
            break
        except OSError:
            # Port in use, try next one
            continue
    
    if port is None:
        pytest.fail(f"Could not find available port for mock server (tried ports {port_candidates[0]}-{port_candidates[-1]})")
    
    # Start server in background thread
    thread = Thread(target=lambda: run_server(app, port), daemon=True)
    thread.start()
    
    # Wait for server to start and verify it's accessible
    # Ensure api_base has trailing slash (LiteLLM appends /v1/chat/completions)
    server_url = f"http://127.0.0.1:{port}/"
    max_attempts = 20  # More attempts to ensure server is ready
    server_ready = False
    for attempt in range(max_attempts):
        time.sleep(0.3)  # Longer wait between attempts
        try:
            # Test the actual endpoint we'll use (LiteLLM appends /v1/chat/completions to api_base)
            response = httpx.post(
                f"{server_url}v1/chat/completions",
                json={"model": TEST_MODEL_NAME, "messages": [{"role": "user", "content": "test"}]},
                timeout=2.0
            )
            if response.status_code == 200:
                server_ready = True
                print(f"[Mock Server] Server ready at {server_url}")
                break
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
            # Server not ready yet, continue waiting
            if attempt == max_attempts - 1:
                pytest.fail(
                    f"Mock server failed to start on {server_url} after {max_attempts} attempts. "
                    f"Could not connect to /v1/chat/completions endpoint. Error: {e}"
                )
            continue
        except Exception as e:
            # Other errors might indicate server is up but endpoint has issues
            # If we get a response (even error), server is running
            print(f"[Mock Server] Server responded with error (but is running): {e}")
            server_ready = True
            break
    
    if not server_ready:
        pytest.fail(f"Mock server not accessible at {server_url} after {max_attempts} attempts")
    
    yield server_url
    
    # Server will be cleaned up when thread dies (daemon=True)


@pytest.fixture
def limit_memory(request):
    """Fixture to track memory usage and enforce limits via @pytest.mark.limit_leaks marker.
    
    Usage:
        @pytest.mark.limit_leaks("40 MB")
        def test_something(limit_memory):
            # Test code here
            # Memory will be measured and test will fail if increase exceeds limit
    """
    marker = request.node.get_closest_marker("limit_leaks")
    if marker:
        # Parse limit from marker (e.g., "30 MB" -> 30)
        limit_str = marker.args[0] if marker.args else "100 MB"
        limit_mb = float(limit_str.split()[0])
        limit_bytes = limit_mb * 1024 * 1024
        
        # Measure baseline memory (router will be fresh from fixture)
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss
        
        yield
        
        # Force GC before measuring final memory
        gc.collect()
        # Small delay for memory to stabilize
        time.sleep(GC_STABILIZATION_DELAY)
        
        # Measure final memory after test
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory
        memory_increase_mb = memory_increase / 1024 / 1024
        
        # Print memory stats
        print(f"\n[Memory Limit Test] Memory usage:")
        print(f"  Baseline: {baseline_memory / 1024 / 1024:.2f} MB")
        print(f"  Final: {final_memory / 1024 / 1024:.2f} MB")
        print(f"  Increase: {memory_increase_mb:+.2f} MB")
        print(f"  Limit: {limit_mb:.2f} MB")
        
        # Fail if memory increase exceeds limit
        if memory_increase > limit_bytes:
            pytest.fail(
                f"Memory limit exceeded: {memory_increase_mb:.2f} MB increase > {limit_mb:.2f} MB limit. "
                f"Baseline: {baseline_memory / 1024 / 1024:.2f} MB, Final: {final_memory / 1024 / 1024:.2f} MB"
            )
    else:
        yield


@pytest.fixture
def test_router(mock_server):
    """Fixture to create a fresh router instance for each test.
    
    Uses the mock server fixture to avoid external API calls.
    Disables cooldowns to prevent deployments from being marked unavailable.
    
    Usage:
        def test_something(test_router, limit_memory):
            # Use test_router for making requests
            response = await test_router.acompletion(...)
    """
    router = Router(
        model_list=[
            {
                "model_name": TEST_MODEL_NAME,
                "litellm_params": {
                    "model": f"openai/{TEST_MODEL_NAME}",
                    "api_base": mock_server,
                    "api_key": TEST_API_KEY,
                },
            },
        ],
        disable_cooldowns=True,  # Disable cooldowns for testing
        allowed_fails=1000,  # Allow many failures before cooldown (effectively disabled)
    )
    yield router
    # Cleanup after test
    try:
        router.discard()
    except Exception:
        pass  # Ignore cleanup errors


async def run_memory_baseline_test(num_requests: int, router: Router, limit_memory):
    """Helper function to run memory baseline test with specified number of requests.
    
    Makes requests concurrently in large batches for speed, optimized for performance.
    
    Args:
        num_requests: Number of requests to make.
        router: Router instance to use for requests.
        limit_memory: Pytest fixture for memory tracking (reference to suppress linter warning).
    
    Example:
        @pytest.mark.asyncio
        @pytest.mark.limit_leaks("40 MB")
        async def test_memory(test_router, limit_memory):
            await run_memory_baseline_test(1000, test_router, limit_memory)
    """
    import asyncio
    
    # Fixture is used automatically by pytest - reference it to suppress linter warning
    _ = limit_memory
    
    # Track memory throughout test
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Memory] Test start: {start_memory:.2f} MB")
    
    # Make requests concurrently in large batches for speed
    # Large batch size = faster tests (temporary memory spike is accounted for in limit)
    BATCH_SIZE = 100
    
    batch_num = 0
    total_batches = (num_requests + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_start in range(0, num_requests, BATCH_SIZE):
        batch_num += 1
        batch_end = min(batch_start + BATCH_SIZE, num_requests)
        
        # Create tasks directly for maximum speed
        tasks = [
            router.acompletion(
                model=TEST_MODEL_NAME,
                messages=[{"role": "user", "content": f"Test request {i}"}],
            )
            for i in range(batch_start, batch_end)
        ]
        
        print(f"[Batch {batch_num}/{total_batches}] Executing batch {batch_start}-{batch_end}...")
        batch_start_time = time.time()
        
        # Execute batch concurrently (return_exceptions=True to not fail on individual errors)
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        batch_duration = time.time() - batch_start_time
        
        # Clean up batch immediately to avoid memory accumulation
        if responses:
            for resp in responses:
                del resp
        del responses
        del tasks
        
        # Periodic memory check every 10 batches
        if batch_num % 10 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"[Batch {batch_num}/{total_batches}] Completed in {batch_duration:.2f}s ({BATCH_SIZE/batch_duration:.1f} req/s) | Memory: {current_memory:.2f} MB (+{current_memory - start_memory:.2f} MB)")
        else:
            print(f"[Batch {batch_num}/{total_batches}] Completed in {batch_duration:.2f}s ({BATCH_SIZE/batch_duration:.1f} req/s)")
    
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
    for i in range(5):  # 5 passes to be thorough
        collected = gc.collect()
        collected_total += collected
        if collected > 0:
            print(f"[Cleanup] Full GC pass {i+1}: collected {collected} objects")
        else:
            print(f"[Cleanup] Full GC pass {i+1}: no objects to collect (clean!)")
    
    # Step 3: Check for uncollectable objects (memory leaks!)
    uncollectable = len(gc.garbage)
    if uncollectable > 0:
        print(f"[Cleanup] WARNING: {uncollectable} uncollectable objects found (potential leak!)")
        print(f"[Cleanup] Uncollectable types: {set(type(obj).__name__ for obj in gc.garbage[:10])}")
    else:
        print(f"[Cleanup] No uncollectable objects (no circular reference leaks)")
    
    # Step 4: Wait for OS to release memory
    print("[Cleanup] Waiting for OS to release memory...")
    time.sleep(0.5)
    
    # Step 5: Final GC to catch anything released during wait
    final_gc = gc.collect()
    collected_total += final_gc
    if final_gc > 0:
        print(f"[Cleanup] Final GC: collected {final_gc} additional objects")
    
    # Final memory check
    memory_after_cleanup = process.memory_info().rss / 1024 / 1024
    print(f"\n[Memory] After cleanup: {memory_after_cleanup:.2f} MB (+{memory_after_cleanup - start_memory:.2f} MB from start)")
    print(f"[Memory] Cleanup freed: {memory_before_cleanup - memory_after_cleanup:.2f} MB")
    print(f"[Memory] Total objects collected: {collected_total:,}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"[Router Memory Test] Completed {num_requests:,} requests")
    print(f"[Router Memory Test] Actual router memory usage: {memory_after_cleanup - start_memory:.2f} MB")
    print(f"[Router Memory Test] Test artifacts cleaned: {memory_before_cleanup - memory_after_cleanup:.2f} MB")
    print(f"{'='*80}\n")
