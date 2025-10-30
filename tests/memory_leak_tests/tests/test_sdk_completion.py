"""
Memory leak tests for LiteLLM SDK completion (sync/async, streaming/non-streaming).
"""

import sys
import os

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../../.."))

import litellm

from .. import (
    run_memory_measurement_with_tracemalloc,
    analyze_and_detect_leaks,
    verify_module_id_consistency,
    get_completion_kwargs,
    get_completion_function,
    get_memory_test_config,
    print_test_header,
)
from ..constants import FAKE_LLM_ENDPOINT


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
    # Get initial module ID and configuration
    initial_id = id(litellm)
    print(f"\n[TEST] litellm module ID: {initial_id}", flush=True)
    verify_module_id_consistency(litellm, initial_id, "at test start")
    
    config = get_memory_test_config()
    print_test_header(title=test_title)
    completion_func = get_completion_function(litellm, use_async, streaming)
    
    # Run measurement phase with tracemalloc
    memory_samples, error_counts = await run_memory_measurement_with_tracemalloc(
        completion_func=completion_func,
        completion_kwargs=get_completion_kwargs(api_base=FAKE_LLM_ENDPOINT),
        config=config,
        module_to_verify=litellm,
        module_id=initial_id,
        test_name=test_title  # Use the test title as the test name for organization
    )
    
    # Analyze results and detect leaks
    analyze_and_detect_leaks(
        memory_samples=memory_samples,
        error_counts=error_counts,
        config=config,
        module_to_verify=litellm,
        module_id=initial_id,
        module_name="Module",
        test_name=test_title  # Pass test name for leak source analysis
    )
