"""
Helper functions for memory leak testing in LiteLLM.

This module provides reusable utilities for detecting memory leaks using tracemalloc,
including warmup phases, measurement, rolling average smoothing, and leak detection.
"""

import gc
import time
import statistics
from typing import List, Dict, Any, Tuple, Optional
import pytest

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_BATCHES,
    DEFAULT_WARMUP_BATCHES,
    DEFAULT_ROLLING_AVERAGE_WINDOW,
    DEFAULT_TEST_MAX_GROWTH_PERCENT,
    DEFAULT_TEST_STABILIZATION_TOLERANCE_MB,
    DEFAULT_NUM_SAMPLES_FOR_GROWTH_ANALYSIS,
    DEFAULT_ERROR_MEMORY_SPIKE_THRESHOLD_PERCENT,
    DEFAULT_ERROR_SPIKE_STABILIZATION_TOLERANCE_PERCENT,
    DEFAULT_ERROR_SPIKE_MIN_STABLE_BATCHES,
    DEFAULT_LEAK_DETECTION_MAX_GROWTH_PERCENT,
    DEFAULT_LEAK_DETECTION_STABILIZATION_TOLERANCE_MB,
    DEFAULT_LEAK_DETECTION_TAIL_SAMPLES,
    MAX_NUM_BATCHES,
    NEAR_ZERO_MEMORY_THRESHOLD_MB,
    DEFAULT_SDK_MODEL,
    DEFAULT_TEST_MESSAGE_CONTENT,
    TEST_API_KEY,
    DEFAULT_TEST_USER,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_REQUEST_PATH,
    DEFAULT_REQUEST_SCHEME,
    DEFAULT_REQUEST_SERVER,
    DEFAULT_REQUEST_CLIENT,
    MOCK_USER_SPEND,
    MOCK_USER_MAX_BUDGET,
    MOCK_TEAM_SPEND,
    MOCK_TEAM_MAX_BUDGET,
    MOCK_TEAM_METADATA,
)


def force_gc() -> None:
    """
    Run garbage collection three times with delays to ensure full cleanup.
    
    This helps ensure that all unreferenced objects are properly collected
    before measuring memory, reducing noise in memory measurements.
    """
    gc.collect()
    time.sleep(0.2)
    gc.collect()
    time.sleep(0.2)
    gc.collect()
    time.sleep(0.2)


def cleanup_litellm_state(litellm_module) -> None:
    """
    Clean up litellm module state and reload it to ensure test isolation.
    
    This function mirrors the cleanup pattern used in other litellm test suites
    (see tests/local_testing/conftest.py, tests/test_litellm/conftest.py).
    
    Steps:
    1. Flush HTTP client caches (critical for memory leak tests)
    2. Clear logging worker queue
    3. Reload module to reset all state (callbacks, caches, sessions)
    4. Force garbage collection
    
    Args:
        litellm_module: The litellm module instance to clean up
    """
    import asyncio
    import importlib
    
    # Step 1: Flush in-memory HTTP client cache
    # This is critical - prevents accumulation of HTTP clients across tests
    if hasattr(litellm_module, 'in_memory_llm_clients_cache'):
        try:
            litellm_module.in_memory_llm_clients_cache.flush_cache()
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not flush client cache: {e}", flush=True)
    
    # Step 2: Clear logging worker queue (prevents log accumulation)
    if hasattr(litellm_module, 'litellm_core_utils'):
        try:
            from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
            asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not clear logging queue: {e}", flush=True)
    
    # Step 3: Force garbage collection before reload
    # Ensures old objects are cleaned up before module reloads
    force_gc()
    
    # Step 4: Reload the module to get a completely fresh state
    # This resets all module-level state:
    # - Callbacks (success_callback, failure_callback, etc.)
    # - Caches (cache, in_memory_llm_clients_cache.cache_dict)
    # - Client sessions (client_session, aclient_session)
    # - All global configuration variables
    try:
        importlib.reload(litellm_module)
    except Exception as e:
        print(f"[CLEANUP] Error: Module reload failed: {e}", flush=True)
        raise  # Fail loudly if reload fails - this is critical for test isolation
    
    # Step 5: Force garbage collection after reload
    # Ensures old module state is fully cleaned up
    force_gc()


def verify_module_id_consistency(module, expected_id: int, stage: str = "") -> None:
    """
    Verify that a module's ID hasn't changed during test execution.
    
    Args:
        module: The module instance to check
        expected_id: The expected ID value
        stage: Optional description of when this check is being performed
    
    Raises:
        AssertionError: If the module ID doesn't match expected_id
    """
    current_id = id(module)
    stage_info = f" {stage}" if stage else ""
    assert current_id == expected_id, (
        f"Module ID changed{stage_info}! Expected {expected_id}, got {current_id}"
    )


def get_completion_kwargs(
    model: str = DEFAULT_SDK_MODEL,
    content: str = DEFAULT_TEST_MESSAGE_CONTENT,
    api_base: str = "http://0.0.0.0:4000",
    api_key: str = TEST_API_KEY,
    user: str = DEFAULT_TEST_USER
) -> Dict[str, Any]:
    """
    Create standardized completion kwargs for memory leak tests.
    
    Args:
        model: The model to use for completion
        content: The message content
        api_base: The API base URL
        api_key: The API key
        user: The user identifier
    
    Returns:
        dict: Completion parameters for test requests
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "api_base": api_base,
        "api_key": api_key,
        "user": user,
    }


def get_completion_function(litellm_module, use_async: bool, streaming: bool):
    """
    Get the appropriate completion function based on async and streaming parameters.
    
    Args:
        litellm_module: The litellm module instance
        use_async: Whether to use async completion (acompletion) or sync (completion)
        streaming: Whether to use streaming mode
    
    Returns:
        Callable: The appropriate completion function or wrapper
    """
    # Define streaming wrapper functions
    async def async_streaming_wrapper(**kwargs):
        response = await litellm_module.acompletion(**kwargs, stream=True)
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    _ = choice.delta.content
        return True

    def sync_streaming_wrapper(**kwargs):
        response = litellm_module.completion(**kwargs, stream=True)
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    _ = choice.delta.content
        return True

    # Select appropriate completion function
    completion_funcs = {
        (True, True): async_streaming_wrapper,
        (True, False): litellm_module.acompletion,
        (False, True): sync_streaming_wrapper,
        (False, False): litellm_module.completion,
    }
    return completion_funcs[(use_async, streaming)]


def get_router_completion_kwargs(
    model: str = DEFAULT_ROUTER_MODEL,
    content: str = DEFAULT_TEST_MESSAGE_CONTENT,
    user: str = DEFAULT_TEST_USER
) -> Dict[str, Any]:
    """
    Create standardized completion kwargs for Router memory leak tests.
    
    Router tests don't need api_base or api_key since the Router instance
    handles those internally based on its model_list configuration.
    
    Args:
        model: The model name (as configured in Router's model_list)
        content: The message content
        user: The user identifier
    
    Returns:
        dict: Completion parameters for Router test requests
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "user": user,
    }


def get_router_completion_function(router, use_async: bool, streaming: bool):
    """
    Get the appropriate router completion function based on async and streaming parameters.
    
    Args:
        router: The Router instance
        use_async: Whether to use async completion (acompletion) or sync (completion)
        streaming: Whether to use streaming mode
    
    Returns:
        Callable: The appropriate completion function or wrapper
    """
    # Define streaming wrapper functions
    async def async_streaming_wrapper(**kwargs):
        response = await router.acompletion(**kwargs, stream=True)
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    _ = choice.delta.content
        return True

    def sync_streaming_wrapper(**kwargs):
        response = router.completion(**kwargs, stream=True)
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    _ = choice.delta.content
        return True

    # Select appropriate completion function
    completion_funcs = {
        (True, True): async_streaming_wrapper,
        (True, False): router.acompletion,
        (False, True): sync_streaming_wrapper,
        (False, False): router.completion,
    }
    return completion_funcs[(use_async, streaming)]


def bytes_to_mb(bytes_value: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_value / (1024 * 1024)


async def run_warmup_phase(
    batch_size: int,
    warmup_batches: int,
    completion_func,
    completion_kwargs: Dict[str, Any]
) -> None:
    """
    Run warmup phase to stabilize caches and memory allocations.
    
    Args:
        batch_size: Number of requests per batch
        warmup_batches: Number of warmup batches to run
        completion_func: The completion function to call (e.g., litellm.acompletion)
        completion_kwargs: Keyword arguments to pass to completion_func
        
    Note:
        Errors during warmup are logged but don't fail the test. This ensures
        transient errors (e.g., network issues) don't prevent memory leak detection.
        Supports both sync and async completion functions.
    """
    import inspect
    
    print("Warming up...")
    error_count = 0
    is_async = inspect.iscoroutinefunction(completion_func)
    
    for batch_idx in range(warmup_batches):
        force_gc()
        for req_idx in range(batch_size):
            try:
                if is_async:
                    response = await completion_func(**completion_kwargs)
                else:
                    response = completion_func(**completion_kwargs)
                assert response is not None, "Completion returned None"
                # Check for error indicators in response
                if hasattr(response, 'error') and response.error:
                    error_count += 1
                    print(f"[WARMUP WARNING] Request {req_idx + 1} in batch {batch_idx + 1} returned error: {response.error} (continuing...)")
                del response
            except Exception as e:
                error_count += 1
                print(f"[WARMUP WARNING] Request {req_idx + 1} in batch {batch_idx + 1} failed: {str(e)} (continuing...)")
                # Continue to next request
                continue
        force_gc()
    
    if error_count > 0:
        print(f"\n[WARMUP INFO] Completed with {error_count} errors (non-fatal, continuing with test)")
    print("Warm-up complete. Starting measured phase...\n")


async def run_measurement_phase(
    batch_size: int,
    num_batches: int,
    completion_func,
    completion_kwargs: Dict[str, Any],
    tracemalloc_module,
    litellm_module=None
) -> Tuple[List[float], List[int]]:
    """
    Run measurement phase and collect memory samples after each batch.
    
    Args:
        batch_size: Number of requests per batch
        num_batches: Number of batches to measure
        completion_func: The completion function to call
        completion_kwargs: Keyword arguments to pass to completion_func
        tracemalloc_module: The tracemalloc module for memory tracking
        litellm_module: Optional litellm module for state cleanup
        
    Returns:
        Tuple of:
            - List of memory measurements in MB for each batch
            - List of error counts for each batch
        
    Note:
        Errors during measurement are logged but don't fail the test. This ensures
        transient errors (e.g., network issues) don't prevent memory leak detection.
        The test continues with remaining requests to collect memory data.
        Supports both sync and async completion functions.
    """
    import inspect
    
    memory_samples = []
    error_counts = []
    total_errors = 0
    is_async = inspect.iscoroutinefunction(completion_func)
    
    for batch in range(num_batches):
        force_gc()
        batch_errors = 0
        
        for req_idx in range(batch_size):
            try:
                if is_async:
                    response = await completion_func(**completion_kwargs)
                else:
                    response = completion_func(**completion_kwargs)
                assert response is not None, "Completion returned None"
                # Check for error indicators in response
                if hasattr(response, 'error') and response.error:
                    batch_errors += 1
                    total_errors += 1
                    if batch_errors <= 3:  # Only print first 3 errors per batch to avoid spam
                        print(f"[BATCH {batch + 1} WARNING] Request {req_idx + 1} returned error: {response.error} (continuing...)")
                del response
            except Exception as e:
                batch_errors += 1
                total_errors += 1
                if batch_errors <= 3:  # Only print first 3 errors per batch to avoid spam
                    print(f"[BATCH {batch + 1} WARNING] Request {req_idx + 1} failed: {str(e)} (continuing...)")
                # Continue to next request
                continue
        
        force_gc()
        
        current, peak = tracemalloc_module.get_traced_memory()
        current_mb = bytes_to_mb(current)
        peak_mb = bytes_to_mb(peak)
        memory_samples.append(current_mb)
        error_counts.append(batch_errors)
        
        error_suffix = f" | Errors: {batch_errors}" if batch_errors > 0 else ""
        print(f"Batch {batch + 1}/{num_batches}: "
              f"Current={current_mb:.2f} MB | Peak={peak_mb:.2f} MB{error_suffix}")
    
    if total_errors > 0:
        print(f"\n[MEASUREMENT INFO] Completed with {total_errors} total errors (non-fatal, memory data collected)")
    
    return memory_samples, error_counts


def calculate_rolling_average(
    memory_samples: List[float],
    sample_window: int
) -> List[float]:
    """
    Calculate rolling average to smooth out memory measurement noise.
    
    Args:
        memory_samples: List of memory measurements in MB
        sample_window: Window size for rolling average
        
    Returns:
        List of smoothed memory values using rolling average
    """
    rolling = [
        statistics.mean(memory_samples[i - sample_window:i])
        for i in range(sample_window, len(memory_samples))
    ]
    return rolling


def analyze_memory_growth(
    rolling_avg: List[float],
    num_samples_for_avg: int = DEFAULT_NUM_SAMPLES_FOR_GROWTH_ANALYSIS
) -> Dict[str, float]:
    """
    Analyze memory growth from smoothed samples.
    
    Args:
        rolling_avg: List of smoothed memory values
        num_samples_for_avg: Number of samples to average at start and end
        
    Returns:
        Dictionary with 'initial_avg', 'final_avg', 'growth', and 'growth_percent'
    """
    initial_avg = statistics.mean(rolling_avg[:num_samples_for_avg])
    final_avg = statistics.mean(rolling_avg[-num_samples_for_avg:])
    growth = final_avg - initial_avg
    growth_percent = (growth / initial_avg * 100) if initial_avg > 0 else 0
    
    return {
        'initial_avg': initial_avg,
        'final_avg': final_avg,
        'growth': growth,
        'growth_percent': growth_percent
    }


def detect_error_induced_memory_leak(
    memory_samples: List[float],
    error_counts: List[int],
    error_spike_threshold_percent: float = DEFAULT_ERROR_MEMORY_SPIKE_THRESHOLD_PERCENT,
    stabilization_tolerance_percent: float = DEFAULT_ERROR_SPIKE_STABILIZATION_TOLERANCE_PERCENT,
    min_stable_batches: int = DEFAULT_ERROR_SPIKE_MIN_STABLE_BATCHES
) -> Tuple[bool, str, List[int]]:
    """
    Detect error-induced memory leaks where errors cause memory spikes that don't get released.
    
    This function identifies the pattern where:
    - A batch has 50%+ memory increase from the previous batch
    - That batch has errors
    - Memory stabilizes at the higher level after the error (doesn't continue growing)
    
    Args:
        memory_samples: List of memory measurements in MB for each batch
        error_counts: List of error counts for each batch
        error_spike_threshold_percent: Minimum percent increase to consider a spike (default: 50%)
        stabilization_tolerance_percent: Max percent variation to consider memory stable (default: 10%)
        min_stable_batches: Minimum batches after spike to check for stabilization (default: 2)
        
    Returns:
        Tuple of:
            - bool: Whether error-induced leak was detected
            - str: Detailed message about the findings
            - List[int]: Batch indices where error-induced spikes occurred (1-indexed)
    """
    if len(memory_samples) < 2 or len(error_counts) < 2:
        return False, "", []
    
    error_spike_batches = []
    non_stabilized_spikes = []
    
    for i in range(1, len(memory_samples)):
        prev_memory = memory_samples[i - 1]
        curr_memory = memory_samples[i]
        curr_errors = error_counts[i]
        
        # Skip if previous memory is too small to calculate meaningful percentage
        if prev_memory < NEAR_ZERO_MEMORY_THRESHOLD_MB:
            continue
        
        # Calculate percent increase
        percent_increase = ((curr_memory - prev_memory) / prev_memory) * 100
        
        # Check if this batch has a significant spike AND errors
        if percent_increase >= error_spike_threshold_percent and curr_errors > 0:
            # Now verify that memory stabilized after the spike
            # Check batches after this spike to see if they stayed at similar level
            batches_after_spike = len(memory_samples) - (i + 1)
            
            if batches_after_spike >= min_stable_batches:
                # Check if subsequent batches are stable (within tolerance of spike level)
                subsequent_batches = memory_samples[i + 1:i + 1 + min_stable_batches]
                
                # Check if all subsequent batches are within tolerance of the spike level
                is_stable = True
                max_variation = 0
                for next_memory in subsequent_batches:
                    variation_percent = abs((next_memory - curr_memory) / curr_memory * 100)
                    max_variation = max(max_variation, variation_percent)
                    if variation_percent > stabilization_tolerance_percent:
                        is_stable = False
                
                # Categorize based on stabilization
                if is_stable:
                    error_spike_batches.append(i + 1)  # Convert to 1-indexed for display
                else:
                    # Track spikes that didn't stabilize
                    non_stabilized_spikes.append({
                        'batch': i + 1,
                        'prev_memory': prev_memory,
                        'curr_memory': curr_memory,
                        'percent_increase': percent_increase,
                        'errors': curr_errors,
                        'max_variation': max_variation,
                        'next_batches': subsequent_batches
                    })
            else:
                # Not enough batches after spike to verify stabilization
                # Still flag it but note this in the message
                error_spike_batches.append(i + 1)
    
    # Build message based on what we found
    message_parts = []
    found_stabilized = len(error_spike_batches) > 0
    found_non_stabilized = len(non_stabilized_spikes) > 0
    
    if found_stabilized:
        # Build detailed message for stabilized spikes
        spike_details = []
        for batch_num in error_spike_batches:
            batch_idx = batch_num - 1  # Convert back to 0-indexed
            prev_memory = memory_samples[batch_idx - 1]
            curr_memory = memory_samples[batch_idx]
            percent_increase = ((curr_memory - prev_memory) / prev_memory) * 100
            errors = error_counts[batch_idx]
            
            # Check stabilization info
            batches_after = len(memory_samples) - (batch_idx + 1)
            if batches_after >= min_stable_batches:
                # Calculate average of stable batches
                stable_batches = memory_samples[batch_idx + 1:batch_idx + 1 + min_stable_batches]
                avg_after = sum(stable_batches) / len(stable_batches)
                stabilization_info = f" → stabilized at {avg_after:.2f} MB"
            else:
                stabilization_info = f" (insufficient batches after spike to confirm stabilization)"
            
            spike_details.append(
                f"  • Batch {batch_num}: {prev_memory:.2f} MB → {curr_memory:.2f} MB "
                f"(+{percent_increase:.1f}%) with {errors} error(s){stabilization_info}"
            )
        
        message_parts.append(
            f"ERROR-INDUCED MEMORY LEAK DETECTED\n"
            f"\n"
            f"Memory spikes occurred in batch(es) with errors and did not fully recover:\n"
            f"{chr(10).join(spike_details)}\n"
            f"\n"
            f"This indicates that error handling is not properly releasing resources.\n"
            f"Check error paths for unreleased connections, buffers, or cached data."
        )
    
    if found_non_stabilized:
        # Add information about non-stabilized spikes
        non_stable_details = []
        for spike in non_stabilized_spikes:
            next_mems = ", ".join([f"{m:.2f}" for m in spike['next_batches'][:3]])
            non_stable_details.append(
                f"  • Batch {spike['batch']}: {spike['prev_memory']:.2f} MB → {spike['curr_memory']:.2f} MB "
                f"(+{spike['percent_increase']:.1f}%) with {spike['errors']} error(s)\n"
                f"    Next batches: {next_mems} MB (variation {spike['max_variation']:.1f}%, did NOT stabilize)"
            )
        
        if found_stabilized:
            message_parts.append(
                f"\n"
                f"NOTE: Additional error spikes detected but memory did NOT stabilize:\n"
                f"{chr(10).join(non_stable_details)}\n"
                f"\n"
                f"These spikes show continued growth after the error, suggesting a continuous\n"
                f"memory leak pattern rather than error-induced stabilization."
            )
        else:
            message_parts.append(
                f"Error spikes detected with continued growth:\n"
                f"{chr(10).join(non_stable_details)}\n"
                f"\n"
                f"Memory continues to grow after error rather than stabilizing.\n"
                f"This suggests a continuous memory leak pattern that may have been\n"
                f"triggered by the error. Check both error paths AND ongoing operations."
            )
    
    if not found_stabilized and not found_non_stabilized:
        return False, "", []
    
    message = "\n".join(message_parts)
    
    # Only return True if we found stabilized spikes (the true error-induced pattern)
    return found_stabilized, message, error_spike_batches


def detect_memory_leak(
    growth_metrics: Dict[str, float],
    memory_samples: List[float],
    error_counts: Optional[List[int]] = None,
    max_growth_percent: float = DEFAULT_LEAK_DETECTION_MAX_GROWTH_PERCENT,
    stabilization_tolerance_mb: float = DEFAULT_LEAK_DETECTION_STABILIZATION_TOLERANCE_MB,
    tail_samples: int = DEFAULT_LEAK_DETECTION_TAIL_SAMPLES,
    error_spike_threshold_percent: float = DEFAULT_ERROR_MEMORY_SPIKE_THRESHOLD_PERCENT
) -> Tuple[bool, str]:
    """
    Detect memory leaks based on growth metrics and continuous growth patterns.
    
    Args:
        growth_metrics: Dictionary from analyze_memory_growth()
        memory_samples: Original memory samples (not smoothed)
        error_counts: Optional list of error counts per batch for error-induced leak detection
        max_growth_percent: Maximum allowed growth percentage
        stabilization_tolerance_mb: Minimum growth to consider significant (MB)
        tail_samples: Number of final samples to check for continuous growth
        error_spike_threshold_percent: Minimum percent increase to consider an error spike
        
    Returns:
        Tuple of (leak_detected: bool, message: str)
    """
    initial_avg = growth_metrics['initial_avg']
    final_avg = growth_metrics['final_avg']
    growth = growth_metrics['growth']
    growth_percent = growth_metrics['growth_percent']
    
    # Check for error-induced memory leaks first (most specific pattern)
    if error_counts is not None:
        error_leak_detected, error_leak_message, spike_batches = detect_error_induced_memory_leak(
            memory_samples, error_counts, error_spike_threshold_percent
        )
        if error_leak_detected:
            return (True, error_leak_message)
    
    # Handle near-zero memory scenarios (tracemalloc may not track very small allocations)
    # If both initial and final averages are very close to zero, consider it as no growth
    if initial_avg < NEAR_ZERO_MEMORY_THRESHOLD_MB and final_avg < NEAR_ZERO_MEMORY_THRESHOLD_MB:
        # No meaningful memory to track - skip leak detection
        return (False, "Memory measurements near zero — tracemalloc tracking not sufficient for this workload")
    
    # Recalculate growth_percent properly to avoid division by zero issues
    if initial_avg > NEAR_ZERO_MEMORY_THRESHOLD_MB:
        growth_percent = (growth / initial_avg * 100)
    elif growth > NEAR_ZERO_MEMORY_THRESHOLD_MB:
        # If we started near zero but grew substantially, that's a potential leak
        growth_percent = (growth / NEAR_ZERO_MEMORY_THRESHOLD_MB * 100)
    else:
        growth_percent = 0
    
    # Check if overall growth exceeds threshold
    if growth_percent > max_growth_percent:
        return (
            True,
            f"Memory grew by {growth_percent:.1f}% "
            f"(>{max_growth_percent}% threshold) — possible leak"
        )
    
    # Check for continuous growth in final samples
    tail = memory_samples[-tail_samples:]
    if len(tail) >= 2:
        continuous_growth = all(
            (tail[i + 1] - tail[i]) > stabilization_tolerance_mb
            for i in range(len(tail) - 1)
        )
        if continuous_growth:
            return (
                True,
                "Continuous memory growth in final batches — possible leak"
            )
    
    return (False, "Memory stabilized — no leak detected")


def print_analysis_header(title: str = "Memory Growth Analysis") -> None:
    """Print a formatted analysis header."""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)


def print_test_header(title: str = "Memory Leak Detection Test") -> None:
    """Print a formatted test header."""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)


def print_growth_metrics(growth_metrics: Dict[str, float]) -> None:
    """Print formatted growth metrics."""
    print(f"Initial avg: {growth_metrics['initial_avg']:.2f} MB")
    print(f"Final avg:   {growth_metrics['final_avg']:.2f} MB")
    print(f"Growth:      {growth_metrics['growth']:.2f} MB "
          f"({growth_metrics['growth_percent']:.1f}%)")


def print_memory_samples(memory_samples: List[float], num_samples: int = 10) -> None:
    """Print the last N memory samples."""
    samples_str = [f'{m:.2f}MB' for m in memory_samples[-num_samples:]]
    print(f"Samples (last {num_samples}): {samples_str}")
    print("="*70)


def create_fastapi_request(url_path: str = DEFAULT_REQUEST_PATH, auth_header: str = f"Bearer {TEST_API_KEY}"):
    """
    Create a real FastAPI Request object for testing.
    
    Args:
        url_path: The URL path for the request
        auth_header: The authorization header value
        
    Returns:
        A FastAPI Request object
    """
    from fastapi import Request
    
    scope = {
        "type": "http",
        "method": "POST",
        "path": url_path,
        "query_string": b"",
        "headers": [
            (b"authorization", auth_header.encode()),
            (b"content-type", b"application/json"),
        ],
        "scheme": DEFAULT_REQUEST_SCHEME,
        "server": DEFAULT_REQUEST_SERVER,
        "client": DEFAULT_REQUEST_CLIENT,
    }
    
    # Create request with a receive callable
    async def receive():
        return {"type": "http.request", "body": b'{}'}
    
    request = Request(scope, receive=receive)
    return request


def get_memory_test_config(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_batches: int = DEFAULT_NUM_BATCHES,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    sample_window: int = DEFAULT_ROLLING_AVERAGE_WINDOW,
    max_growth_percent: float = DEFAULT_TEST_MAX_GROWTH_PERCENT,
    stabilization_tolerance_mb: float = DEFAULT_TEST_STABILIZATION_TOLERANCE_MB
) -> Dict[str, Any]:
    """
    Get standardized memory test configuration parameters.
    
    Centralized configuration ensures consistent testing across all memory leak tests.
    All defaults are imported from constants module.
    
    Args:
        batch_size: Number of requests per batch
        num_batches: Number of batches to measure (max: MAX_NUM_BATCHES)
        warmup_batches: Number of warmup batches to run
        sample_window: Rolling average window size
        max_growth_percent: Maximum allowed growth percentage
        stabilization_tolerance_mb: Minimum growth to consider significant in MB
    
    Returns:
        dict: Configuration parameters for memory leak tests
    """
    # Enforce maximum num_batches to prevent overly long test runs
    if num_batches > MAX_NUM_BATCHES:
        num_batches = MAX_NUM_BATCHES
    
    return {
        'batch_size': batch_size,
        'num_batches': num_batches,
        'warmup_batches': warmup_batches,
        'sample_window': sample_window,
        'max_growth_percent': max_growth_percent,
        'stabilization_tolerance_mb': stabilization_tolerance_mb
    }


def setup_proxy_server_dependencies(proxy_server_module, mock_prisma_client, real_cache, real_logging):
    """
    Set up proxy_server module dependencies for auth testing.
    
    Args:
        proxy_server_module: The proxy_server module to configure
        mock_prisma_client: Mock PrismaClient instance
        real_cache: Real DualCache instance
        real_logging: Real ProxyLogging instance
    """
    proxy_server_module.prisma_client = mock_prisma_client
    proxy_server_module.user_api_key_cache = real_cache
    proxy_server_module.proxy_logging_obj = real_logging
    proxy_server_module.general_settings = {
        "pass_through_endpoints": None,
        "enable_jwt_auth": False,
        "enable_oauth2_auth": False,
        "enable_oauth2_proxy_auth": False,
        "allow_user_auth": False,
    }
    proxy_server_module.master_key = "different-master-key-to-avoid-matching"
    proxy_server_module.llm_model_list = []
    proxy_server_module.llm_router = None
    proxy_server_module.model_max_budget_limiter = None
    proxy_server_module.open_telemetry_logger = None
    proxy_server_module.litellm_proxy_admin_name = "admin"
    proxy_server_module.user_custom_auth = None
    proxy_server_module.jwt_handler = None
    proxy_server_module.premium_user = False


def save_proxy_server_state(proxy_server_module) -> Dict[str, Any]:
    """
    Save current proxy_server module state for later restoration.
    
    Args:
        proxy_server_module: The proxy_server module to save state from
    
    Returns:
        dict: Saved state values
    """
    return {
        'prisma_client': getattr(proxy_server_module, 'prisma_client', None),
        'user_api_key_cache': getattr(proxy_server_module, 'user_api_key_cache', None),
        'proxy_logging_obj': getattr(proxy_server_module, 'proxy_logging_obj', None),
        'general_settings': getattr(proxy_server_module, 'general_settings', None),
        'master_key': getattr(proxy_server_module, 'master_key', None),
        'llm_model_list': getattr(proxy_server_module, 'llm_model_list', None),
        'llm_router': getattr(proxy_server_module, 'llm_router', None),
        'model_max_budget_limiter': getattr(proxy_server_module, 'model_max_budget_limiter', None),
        'open_telemetry_logger': getattr(proxy_server_module, 'open_telemetry_logger', None),
        'litellm_proxy_admin_name': getattr(proxy_server_module, 'litellm_proxy_admin_name', None),
        'user_custom_auth': getattr(proxy_server_module, 'user_custom_auth', None),
        'jwt_handler': getattr(proxy_server_module, 'jwt_handler', None),
        'premium_user': getattr(proxy_server_module, 'premium_user', None),
    }


def restore_proxy_server_state(proxy_server_module, saved_state: Dict[str, Any]) -> None:
    """
    Restore proxy_server module state from saved values.
    
    Args:
        proxy_server_module: The proxy_server module to restore state to
        saved_state: Dictionary of saved state values
    """
    for key, value in saved_state.items():
        setattr(proxy_server_module, key, value)


def create_mock_prisma_client(api_key: str):
    """
    Create a mock PrismaClient for auth testing.
    
    Args:
        api_key: The API key to use for creating mock responses
    
    Returns:
        Mock PrismaClient instance configured for testing
    """
    from unittest.mock import AsyncMock, MagicMock
    from litellm.proxy._types import hash_token, LiteLLM_VerificationTokenView
    
    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    
    # Create the token hash for our test key
    test_key_hash = hash_token(api_key)
    
    async def mock_get_data(*args, **kwargs):
        """
        Create a NEW object on each call to simulate real DB behavior.
        
        Real implementation:
        1. Executes SQL query
        2. Parses result into new Pydantic LiteLLM_VerificationTokenView object
        3. Returns that new object
        
        If we return the same object every time, we won't detect leaks from:
        - Pydantic object creation
        - model_dump() calls
        - UserAPIKeyAuth conversion
        """
        mock_key_obj = LiteLLM_VerificationTokenView(
            token=test_key_hash,
            key_alias=api_key,
            spend=MOCK_USER_SPEND,
            max_budget=MOCK_USER_MAX_BUDGET,
            user_id=DEFAULT_TEST_USER,
            team_id=DEFAULT_TEST_USER.replace("user", "team"),
            models=[],
            blocked=False,
            metadata={},
            # VerificationTokenView-specific fields (from combined_view query)
            team_spend=MOCK_TEAM_SPEND,
            team_alias="Test Team",
            team_tpm_limit=None,
            team_rpm_limit=None,
            team_max_budget=MOCK_TEAM_MAX_BUDGET,
            team_models=[],
            team_blocked=False,
            team_metadata=MOCK_TEAM_METADATA,
            team_member=None,
            last_refreshed_at=None,
        )
        return mock_key_obj
    
    mock_prisma_client.get_data = AsyncMock(side_effect=mock_get_data)
    
    return mock_prisma_client


def create_auth_call_function(master_key: str = TEST_API_KEY):
    """
    Create a function that calls _user_api_key_auth_builder.
    
    This factory function returns a callable suitable for memory testing.
    Uses the master key for successful auth without DB.
    
    Args:
        master_key: The master key to use for authentication
        
    Returns:
        A callable function that performs auth and returns True on success
    """
    import asyncio
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder
    
    def call_auth():
        """
        Call the actual user_api_key_auth function.
        
        This is the real function we're testing for memory leaks.
        Uses the master key for successful auth without DB.
        Calls _user_api_key_auth_builder directly to bypass FastAPI dependencies.
        """
        request = create_fastapi_request(
            url_path=DEFAULT_REQUEST_PATH,
            auth_header=f"Bearer {master_key}"
        )
        
        async def _async_call():
            result = await _user_api_key_auth_builder(
                request=request,
                api_key=f"Bearer {master_key}",  # Add Bearer prefix
                azure_api_key_header="",  # Empty string, not None
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={},
            )
            # Clean up the result to release references
            del result
            return True
        
        try:
            return asyncio.run(_async_call())
        except Exception as e:
            # Fail the test if auth call fails
            pytest.fail(f"Auth call failed: {e}")
    
    return call_auth