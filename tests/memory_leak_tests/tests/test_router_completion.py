"""
Memory leak tests for LiteLLM Router completion (sync/async, streaming/non-streaming).

This module tests the Router API to detect memory leaks using a fake LLM endpoint.
"""

import sys
import os

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../../.."))

from litellm import Router

from .. import (
    run_memory_measurement_with_tracemalloc,
    analyze_and_detect_leaks,
    verify_module_id_consistency,
    get_router_completion_kwargs,
    get_router_completion_function,
    get_memory_test_config,
    print_test_header,
)
from ..constants import (
    FAKE_LLM_ENDPOINT,
    DEFAULT_ROUTER_TIMEOUT,
    DEFAULT_ROUTER_NUM_RETRIES,
    TEST_API_KEY,
    DEFAULT_SDK_MODEL,
)


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
async def test_router_completion_memory_leak_with_growth_detection(use_async, streaming, test_title):
    """
    Detects memory leaks in Router completion/acompletion using tracemalloc.
    Tests all combinations of sync/async and streaming/non-streaming modes.
    Uses warm-up phase and rolling averages to filter out allocator noise.
    
    Args:
        use_async: Whether to test acompletion (True) or completion (False)
        streaming: Whether to enable streaming mode
        test_title: Display title for this test variant
    """
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
        num_retries=DEFAULT_ROUTER_NUM_RETRIES,
        timeout=DEFAULT_ROUTER_TIMEOUT,
    )
    
    # Get initial router ID and configuration
    initial_id = id(router)
    print(f"\n[TEST] Router instance ID: {initial_id}", flush=True)
    verify_module_id_consistency(router, initial_id, "at test start")
    
    config = get_memory_test_config()
    
    print_test_header(title=test_title)
    completion_func = get_router_completion_function(router, use_async, streaming)
    
    # Run measurement phase with tracemalloc
    memory_samples, error_counts = await run_memory_measurement_with_tracemalloc(
        completion_func=completion_func,
        completion_kwargs=get_router_completion_kwargs(),
        config=config,
        module_to_verify=router,
        module_id=initial_id,
        test_name=test_title  # Use the test title as the test name for organization
    )
    
    # Analyze results and detect leaks
    analyze_and_detect_leaks(
        memory_samples=memory_samples,
        error_counts=error_counts,
        config=config,
        module_to_verify=router,
        module_id=initial_id,
        module_name="Router instance",
        test_name=test_title  # Pass test name for leak source analysis
    )

