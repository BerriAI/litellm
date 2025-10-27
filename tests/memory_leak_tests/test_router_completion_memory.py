"""
Memray-based memory leak test for LiteLLM Router completion functionality.

This module tests the Router API to detect memory leaks using a fake LLM endpoint.
Uses Memray's Python API for programmatic memory tracking and analysis.
"""

import sys
import os
import tracemalloc

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router

from .constants import FAKE_LLM_ENDPOINT, DEFAULT_ROUTER_TIMEOUT, DEFAULT_ROUTER_NUM_RETRIES
from .memory_test_helpers import (
    run_warmup_phase,
    run_measurement_phase,
    calculate_rolling_average,
    analyze_memory_growth,
    detect_memory_leak,
    print_test_header,
    print_analysis_header,
    print_growth_metrics,
    print_memory_samples,
    verify_module_id_consistency,
    get_router_completion_kwargs,
    get_router_completion_function,
    get_memory_test_config,
    force_gc
)


def _clean_router_state(router):
    """
    Helper to clean router state between requests.
    
    Cleans up Router internal state similar to how cleanup_litellm_state works:
    - Flushes HTTP client caches
    - Clears any accumulated state
    - Forces garbage collection
    
    Args:
        router: Router instance to clean
    """
    # Step 1: Flush HTTP client cache if router has one
    # Router instances may have their own client cache or use the global one
    if hasattr(router, 'client_cache'):
        try:
            if hasattr(router.client_cache, 'flush_cache'):
                router.client_cache.flush_cache()
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not flush router client cache: {e}", flush=True)
    
    # Step 2: Clear any router-specific caches
    # Router may have internal caches for model routing decisions
    if hasattr(router, 'cache'):
        try:
            if hasattr(router.cache, 'flush_cache'):
                router.cache.flush_cache()
            elif hasattr(router.cache, 'clear'):
                router.cache.clear()
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not clear router cache: {e}", flush=True)
    
    # Step 3: Force garbage collection to clean up any references
    force_gc()

@pytest.fixture(autouse=True, scope="function")
def cleanup_router():
    """
    Fixture to ensure complete isolation between parameterized test cases.
    
    This fixture runs before and after each test to clean up router state,
    ensuring that one test (e.g., async-streaming) doesn't affect another 
    (e.g., sync-non-streaming).
    
    Note: Unlike the SDK tests that clean up a global module, Router tests
    create a fresh Router instance for each test, so this cleanup is lighter.
    """
    # Cleanup before test
    print("\n[FIXTURE] Preparing for router test...", flush=True)
    force_gc()
    
    # Run the test
    yield
    
    # Cleanup after test
    print("\n[FIXTURE] Cleaning up after router test...", flush=True)
    force_gc()


@pytest.fixture
def clean_router():
    """
    Fixture that provides a clean Router instance for testing.
    
    This fixture creates a Router with a model pointing to a fake LLM endpoint,
    ensuring clean state before and after the test.
    
    Returns:
        A Router instance with cleaned state
    """
    print("\n[FIXTURE] Creating clean router...", flush=True)
    
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
    
    # Clean the router instance before test
    _clean_router_state(router)
    
    # Verify ID hasn't changed after cleaning
    assert id(router) == router_id, "Router instance ID changed after cleaning!"
    
    print("[FIXTURE] Setup complete, yielding router...", flush=True)
    yield router
    print("[FIXTURE] Test done, cleaning up router...", flush=True)
    
    # Verify ID is still the same at teardown
    assert id(router) == router_id, "Router instance ID changed during test execution!"
    
    # Cleanup after test
    _clean_router_state(router)


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
    # Verify we're using the same router instance from the fixture
    initial_id = id(clean_router)
    print(f"\n[TEST] Received clean_router with ID: {initial_id}", flush=True)
    verify_module_id_consistency(clean_router, initial_id, "at test start")
    
    # Get standardized test configuration (all values from constants.py)
    config = get_memory_test_config()

    print_test_header(title=test_title)
    completion_func = get_router_completion_function(clean_router, use_async, streaming)
    
    tracemalloc.start()

    try:
        # --- Warm-up Phase ---
        await run_warmup_phase(
            batch_size=config['batch_size'],
            warmup_batches=config['warmup_batches'],
            completion_func=completion_func,
            completion_kwargs=get_router_completion_kwargs()
        )
        
        # Verify ID hasn't changed after warmup
        verify_module_id_consistency(clean_router, initial_id, "after warmup")

        # --- Measurement Phase ---
        memory_samples = await run_measurement_phase(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            completion_func=completion_func,
            completion_kwargs=get_router_completion_kwargs(),
            tracemalloc_module=tracemalloc,
            litellm_module=None  # Router handles its own state
        )
        
        # Verify ID hasn't changed after measurement
        verify_module_id_consistency(clean_router, initial_id, "after measurement")

    finally:
        tracemalloc.stop()

    # --- Analysis Phase ---
    print_analysis_header()

    if len(memory_samples) < config['sample_window'] * 2:
        pytest.skip("Not enough samples for reliable growth analysis")

    # Calculate dynamic parameters based on sample_window
    rolling_avg = calculate_rolling_average(memory_samples, config['sample_window'])
    # Use 2x the sample_window for averaging (ensures we smooth over enough data)
    num_samples_for_avg = min(config['sample_window'] * 2, len(rolling_avg) // 3)
    # Use 3x the sample_window for tail analysis (detect continuous growth)
    tail_samples = min(config['sample_window'] * 3, len(memory_samples) // 2)
    
    growth_metrics = analyze_memory_growth(rolling_avg, num_samples_for_avg=num_samples_for_avg)
    
    print_growth_metrics(growth_metrics)

    # Detect memory leaks
    leak_detected, message = detect_memory_leak(
        growth_metrics=growth_metrics,
        memory_samples=memory_samples,
        max_growth_percent=config['max_growth_percent'],
        stabilization_tolerance_mb=config['stabilization_tolerance_mb'],
        tail_samples=tail_samples
    )

    if leak_detected:
        pytest.fail(message)
    
    print(message)
    print_memory_samples(memory_samples, num_samples=10)
    
    # Final verification that ID remained consistent throughout
    verify_module_id_consistency(clean_router, initial_id, "at test end")
    print(f"\n[TEST] ✓ Router instance ID remained consistent throughout: {initial_id}", flush=True)

