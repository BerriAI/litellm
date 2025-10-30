"""
Memray-based memory leak test for LiteLLM Router completion functionality.

This module tests the Router API to detect memory leaks using a fake LLM endpoint.
Uses Memray's Python API for programmatic memory tracking and analysis.
"""

import sys
import os

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router

from .constants import FAKE_LLM_ENDPOINT, DEFAULT_ROUTER_TIMEOUT, DEFAULT_ROUTER_NUM_RETRIES
from .memory_test_helpers import (
    run_memory_measurement_with_tracemalloc,
    analyze_and_detect_leaks,
    print_test_header,
    verify_module_id_consistency,
    get_router_completion_kwargs,
    get_router_completion_function,
    get_memory_test_config,
)


@pytest.fixture
def clean_router():
    """
    Fixture that provides a Router instance for testing.
    
    This fixture creates a Router with a model pointing to a fake LLM endpoint.
    Each test gets a fresh Router instance, which provides natural isolation.
    
    Returns:
        A Router instance
    """
    print("\n[FIXTURE] Creating router...", flush=True)
    
    # Import constants for router configuration
    from .constants import TEST_API_KEY, DEFAULT_SDK_MODEL
    
    # Create router with model pointing to fake endpoint
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": DEFAULT_SDK_MODEL,
                    "api_base": FAKE_LLM_ENDPOINT,
                    "api_key": TEST_API_KEY,
                },
            }
        ],
        num_retries=DEFAULT_ROUTER_NUM_RETRIES,  # Disable retries for cleaner testing
        timeout=DEFAULT_ROUTER_TIMEOUT,
    )
    
    # Store the instance ID for verification
    router_id = id(router)
    print(f"[FIXTURE] Router instance ID: {router_id}", flush=True)
    
    print("[FIXTURE] Setup complete, yielding router...", flush=True)
    yield router
    
    # Verify ID is still the same at teardown
    assert id(router) == router_id, "Router instance ID changed during test execution!"
    
    print("[FIXTURE] Test done", flush=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_async,streaming,test_title",
    [
        (True, False, "Router acompletion Memory Leak Detection Test"),
        (True, True, "Router acompletion Streaming Memory Leak Detection Test"),
        (False, False, "Router completion Memory Leak Detection Test"),
        (False, True, "Router completion Streaming Memory Leak Detection Test"),
    ],
    ids=["async-non-streaming", "async-streaming", "sync-non-streaming", "sync-streaming"]
)
async def test_router_completion_memory_leak_with_growth_detection(clean_router, use_async, streaming, test_title):
    """
    Detects memory leaks in Router completion/acompletion using tracemalloc.
    Tests all combinations of sync/async and streaming/non-streaming modes.
    Uses warm-up phase and rolling averages to filter out allocator noise.
    
    Args:
        clean_router: The Router instance with cleaned state (from fixture)
        use_async: Whether to test acompletion (True) or completion (False)
        streaming: Whether to enable streaming mode
        test_title: Display title for this test variant
    """
    # Get initial router ID and configuration
    initial_id = id(clean_router)
    print(f"\n[TEST] Received clean_router with ID: {initial_id}", flush=True)
    verify_module_id_consistency(clean_router, initial_id, "at test start")
    
    config = get_memory_test_config()
    print_test_header(title=test_title)
    completion_func = get_router_completion_function(clean_router, use_async, streaming)
    
    # Run measurement phase with tracemalloc
    memory_samples, error_counts = await run_memory_measurement_with_tracemalloc(
        completion_func=completion_func,
        completion_kwargs=get_router_completion_kwargs(),
        config=config,
        module_to_verify=clean_router,
        module_id=initial_id
    )
    
    # Analyze results and detect leaks
    analyze_and_detect_leaks(
        memory_samples=memory_samples,
        error_counts=error_counts,
        config=config,
        module_to_verify=clean_router,
        module_id=initial_id,
        module_name="Router instance"
    )

