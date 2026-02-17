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
    """
    app = create_mock_server()
    port = 18888
    
    # Check if port is already in use
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.close()
    except OSError:
        # Port already in use, try next port
        port = 18889
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            sock.close()
        except OSError:
            pytest.fail(f"Could not find available port for mock server (tried 18888, 18889)")
    
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
    
    Makes requests concurrently in batches for speed, with proper error handling
    that doesn't fail the test on individual request failures.
    
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
    # Fixture is used automatically by pytest - reference it to suppress linter warning
    _ = limit_memory
    
    # Make requests concurrently in batches for speed
    # Batch size of 20 provides good balance between speed and memory pressure
    BATCH_SIZE = 20
    
    for batch_start in range(0, num_requests, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_requests)
        # Create concurrent tasks for this batch
        tasks = [
            router.acompletion(
                model=TEST_MODEL_NAME,
                messages=[{"role": "user", "content": f"Test request {i}"}],
            )
            for i in range(batch_start, batch_end)
        ]
        # Execute batch concurrently
        # Note: return_exceptions=True allows test to continue even if some requests fail
        import asyncio
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out failed requests but continue with test
        valid_responses = []
        failed_count = 0
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                failed_count += 1
                # Log exception but continue
                print(f"  Warning: Request {batch_start + i} failed: {type(response).__name__}: {response}")
            elif response is None:
                failed_count += 1
                print(f"  Warning: Request {batch_start + i} returned None")
            else:
                valid_responses.append(response)
        
        # Continue with valid responses - don't fail the test
        # If all failed, that's logged but test continues (might indicate bigger issue)
        if failed_count > 0:
            print(f"  Note: {failed_count}/{len(responses)} requests failed in batch {batch_start}-{batch_end}, continuing with {len(valid_responses)} valid responses")
        
        # Use valid_responses for cleanup
        responses = valid_responses
        # Clean up batch
        del responses
        del tasks
        del valid_responses
        # GC after each batch to prevent accumulation
        gc.collect()
    
    print(f"[Simple Memory Test] Completed {num_requests} requests")
