"""
Memory leak tests for LiteLLM SDK completion (sync/async, streaming/non-streaming).
"""

import sys
import os
import tracemalloc

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../.."))

import litellm

from .constants import FAKE_LLM_ENDPOINT
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
    get_completion_kwargs,
    get_completion_function,
    get_memory_test_config,
    cleanup_litellm_state
)


@pytest.fixture(autouse=True, scope="function")
def cleanup_between_tests():
    """
    Fixture to ensure complete isolation between parameterized test cases.
    
    This fixture runs before and after each test to clean up litellm module state,
    ensuring that one test (e.g., async-streaming) doesn't affect another 
    (e.g., sync-non-streaming).
    
    Cleans up:
    - HTTP client caches
    - Response caches  
    - Client sessions
    - Callbacks
    """
    # Cleanup before test
    print("\n[FIXTURE] Cleaning up litellm state before test...", flush=True)
    cleanup_litellm_state(litellm)
    
    # Run the test
    yield
    
    # Cleanup after test
    print("\n[FIXTURE] Cleaning up litellm state after test...", flush=True)
    cleanup_litellm_state(litellm)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_async,streaming,test_title",
    [
        (True, False, "SDK acompletion Memory Leak Detection Test"),
        (True, True, "SDK acompletion Streaming Memory Leak Detection Test"),
        (False, False, "SDK completion Memory Leak Detection Test"),
        (False, True, "SDK completion Streaming Memory Leak Detection Test"),
    ],
    ids=["async-non-streaming", "async-streaming", "sync-non-streaming", "sync-streaming"]
)
async def test_completion_memory_leak_with_growth_detection(use_async, streaming, test_title):
    """
    Detects memory leaks in completion/acompletion using tracemalloc.
    Tests all combinations of sync/async and streaming/non-streaming modes.
    Uses warm-up phase and rolling averages to filter out allocator noise.
    
    Args:
        use_async: Whether to test acompletion (True) or completion (False)
        streaming: Whether to enable streaming mode
        test_title: Display title for this test variant
    """
    initial_id = id(litellm)
    print(f"\n[TEST] litellm module ID: {initial_id}", flush=True)
    verify_module_id_consistency(litellm, initial_id, "at test start")
    
    config = get_memory_test_config()
    print_test_header(title=test_title)
    completion_func = get_completion_function(litellm, use_async, streaming)
    
    tracemalloc.start()

    try:
        await run_warmup_phase(
            batch_size=config['batch_size'],
            warmup_batches=config['warmup_batches'],
            completion_func=completion_func,
            completion_kwargs=get_completion_kwargs(api_base=FAKE_LLM_ENDPOINT)
        )
        verify_module_id_consistency(litellm, initial_id, "after warmup")

        memory_samples = await run_measurement_phase(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            completion_func=completion_func,
            completion_kwargs=get_completion_kwargs(api_base=FAKE_LLM_ENDPOINT),
            tracemalloc_module=tracemalloc,
            litellm_module=litellm
        )
        verify_module_id_consistency(litellm, initial_id, "after measurement")

    finally:
        tracemalloc.stop()

    print_analysis_header()

    if len(memory_samples) < config['sample_window'] * 2:
        pytest.skip("Not enough samples for reliable growth analysis")

    rolling_avg = calculate_rolling_average(memory_samples, config['sample_window'])
    num_samples_for_avg = min(config['sample_window'] * 2, len(rolling_avg) // 3)
    tail_samples = min(config['sample_window'] * 3, len(memory_samples) // 2)
    
    growth_metrics = analyze_memory_growth(rolling_avg, num_samples_for_avg=num_samples_for_avg)
    print_growth_metrics(growth_metrics)

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
    
    verify_module_id_consistency(litellm, initial_id, "at test end")
    print(f"\n[TEST] ✓ Module ID remained consistent throughout: {initial_id}", flush=True)
