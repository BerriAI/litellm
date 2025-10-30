"""
Test execution and orchestration for memory leak testing.

Provides high-level functions for:
- Executing single requests with error handling
- Running warmup phases
- Running measurement phases
- Orchestrating complete test runs
- Analyzing results and detecting leaks
"""

from typing import List, Dict, Any, Tuple, Optional
import pytest

from ..analysis.detection import detect_memory_leak
from ..analysis.growth import (
    prepare_memory_analysis,
    analyze_memory_growth,
)
from ..analysis.noise import check_measurement_noise
from ..constants import (
    DEFAULT_CAPTURE_TOP_CONSUMERS,
    DEFAULT_TOP_CONSUMERS_COUNT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT,
    DEFAULT_PERIODIC_SAMPLE_INTERVAL,
)
from ..snapshot.capture import (
    should_capture_detailed_snapshot,
    capture_request_memory_snapshot,
)
from ..snapshot.storage import (
    save_batch_snapshots,
    print_final_snapshot_summary,
)
from ..utils.conversions import bytes_to_mb
from ..utils.formatting import (
    print_analysis_header,
    print_growth_metrics,
    print_memory_samples,
)
from .cleanup import (
    force_gc,
    verify_module_id_consistency,
)


async def execute_single_request(
    completion_func,
    completion_kwargs: Dict[str, Any],
    is_async: bool,
    batch: int,
    req_idx: int,
    max_errors_to_print: int = 3,
    current_batch_errors: int = 0
) -> Tuple[bool, Optional[str]]:
    """
    Execute a single completion request and handle any errors.
    
    Args:
        completion_func: The completion function to call
        completion_kwargs: Keyword arguments to pass to completion_func
        is_async: Whether the completion function is async
        batch: Current batch index (0-indexed)
        req_idx: Request index within the batch (0-indexed)
        max_errors_to_print: Maximum number of errors to print per batch
        current_batch_errors: Current error count for this batch
        
    Returns:
        Tuple of (request_had_error: bool, error_message: Optional[str])
    """
    try:
        if is_async:
            response = await completion_func(**completion_kwargs)
        else:
            response = completion_func(**completion_kwargs)
        
        assert response is not None, "Completion returned None"
        
        # Check for error indicators in response
        if hasattr(response, 'error') and response.error:
            error_message = str(response.error)
            if current_batch_errors < max_errors_to_print:
                print(f"[BATCH {batch + 1} WARNING] Request {req_idx + 1} returned error: {response.error} (continuing...)")
            del response
            return True, error_message
        
        del response
        return False, None
        
    except Exception as e:
        error_message = str(e)
        if current_batch_errors < max_errors_to_print:
            print(f"[BATCH {batch + 1} WARNING] Request {req_idx + 1} failed: {str(e)} (continuing...)")
        return True, error_message


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
    capture_top_consumers: bool = DEFAULT_CAPTURE_TOP_CONSUMERS,
    top_consumers_count: int = DEFAULT_TOP_CONSUMERS_COUNT,
    output_dir: Optional[str] = DEFAULT_OUTPUT_DIR,
    smart_capture: bool = True,
    memory_increase_threshold_pct: float = DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT,
    periodic_sample_interval: int = DEFAULT_PERIODIC_SAMPLE_INTERVAL,
    test_name: str = "memory_test"
) -> Tuple[List[float], List[int]]:
    """
    Run measurement phase and collect memory samples after each batch.
    
    Args:
        batch_size: Number of requests per batch
        num_batches: Number of batches to measure
        completion_func: The completion function to call
        completion_kwargs: Keyword arguments to pass to completion_func
        tracemalloc_module: The tracemalloc module for memory tracking
        capture_top_consumers: If True, capture top memory consumers after each request
        top_consumers_count: Number of top consumers to capture
        output_dir: Directory to store snapshot JSON files (one file per test)
        smart_capture: If True, only capture detailed top consumers on memory increases or errors
        memory_increase_threshold_pct: Percentage increase to trigger detailed snapshot capture
        periodic_sample_interval: Capture detailed snapshot every N requests
        test_name: Name of the test (used to create snapshot filename)
        
    Returns:
        Tuple of:
            - List of memory measurements in MB for each batch
            - List of error counts for each batch
        
    Note:
        Errors during measurement are logged but don't fail the test. This ensures
        transient errors (e.g., network issues) don't prevent memory leak detection.
        The test continues with remaining requests to collect memory data.
        Supports both sync and async completion functions.
        
        When capture_top_consumers is True, memory snapshots are taken after each
        individual request and buffered in memory for that batch, then written to disk
        after each batch completes. This prevents:
        - Memory buildup from holding all snapshots until the end (avoiding false positives)
        - Data loss if the test crashes before completion
        - Excessive I/O overhead from writing after every request
        
        When smart_capture is True (default), detailed top consumer data is only captured
        when memory increases significantly (>2%) or when errors occur. This dramatically
        reduces overhead and file size while still capturing important events.
        
        Each test gets its own JSON file in output_dir, making parallel test execution
        safe and avoiding file locking issues.
    """
    import inspect
    
    memory_samples = []
    error_counts = []
    total_errors = 0
    is_async = inspect.iscoroutinefunction(completion_func)
    global_request_counter = 0
    
    # Buffer for memory snapshots - written after each batch to prevent memory buildup
    snapshot_buffer = []
    last_captured_memory_mb = 0.0
    
    for batch in range(num_batches):
        force_gc()
        batch_errors = 0
        
        # Clear the buffer at the start of each batch
        snapshot_buffer.clear()
        
        for req_idx in range(batch_size):
            global_request_counter += 1
            
            # Execute request and handle errors
            request_had_error, error_message = await execute_single_request(
                completion_func=completion_func,
                completion_kwargs=completion_kwargs,
                is_async=is_async,
                batch=batch,
                req_idx=req_idx,
                current_batch_errors=batch_errors
            )
            
            if request_had_error:
                batch_errors += 1
                total_errors += 1
            
            # Capture memory snapshots if requested (buffered in memory for this batch)
            if capture_top_consumers:
                current, _ = tracemalloc_module.get_traced_memory()
                current_mb = bytes_to_mb(current)
                
                # Determine if we should capture detailed or lightweight snapshot
                capture_detailed = True
                if smart_capture:
                    capture_detailed = should_capture_detailed_snapshot(
                        global_request_counter=global_request_counter,
                        request_had_error=request_had_error,
                        current_mb=current_mb,
                        last_captured_memory_mb=last_captured_memory_mb,
                        memory_increase_threshold_pct=memory_increase_threshold_pct,
                        periodic_sample_interval=periodic_sample_interval
                    )
                
                snapshot_data = capture_request_memory_snapshot(
                    tracemalloc_module=tracemalloc_module,
                    global_request_counter=global_request_counter,
                    batch=batch,
                    req_idx=req_idx,
                    request_had_error=request_had_error,
                    error_message=error_message,
                    top_consumers_count=top_consumers_count,
                    lightweight=not capture_detailed
                )
                snapshot_buffer.append(snapshot_data)
                
                if capture_detailed:
                    last_captured_memory_mb = current_mb
        
        # Write snapshots for this batch to disk before continuing
        # This prevents memory buildup and ensures data is saved even if test crashes
        if capture_top_consumers and output_dir:
            save_batch_snapshots(
                snapshot_buffer=snapshot_buffer,
                output_dir=output_dir,
                test_name=test_name
            )
        
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
    
    # Print final summary of captured snapshots
    if capture_top_consumers and output_dir:
        print_final_snapshot_summary(
            output_dir=output_dir,
            test_name=test_name,
            smart_capture=smart_capture
        )
    
    return memory_samples, error_counts


async def run_memory_measurement_with_tracemalloc(
    completion_func,
    completion_kwargs: Dict[str, Any],
    config: Dict[str, Any],
    module_to_verify=None,
    module_id: Optional[int] = None,
    capture_top_consumers: bool = DEFAULT_CAPTURE_TOP_CONSUMERS,
    top_consumers_count: int = DEFAULT_TOP_CONSUMERS_COUNT,
    output_dir: Optional[str] = DEFAULT_OUTPUT_DIR,
    smart_capture: bool = True,
    memory_increase_threshold_pct: float = DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT,
    periodic_sample_interval: int = DEFAULT_PERIODIC_SAMPLE_INTERVAL,
    test_name: str = "memory_test"
) -> Tuple[List[float], List[int]]:
    """
    Run warmup and measurement phases with tracemalloc tracking.
    
    This is a high-level helper that orchestrates the entire measurement process:
    1. Start tracemalloc
    2. Run warmup phase
    3. Run measurement phase
    4. Stop tracemalloc
    5. Optionally verify module ID consistency
    6. Optionally capture top memory consumers and save to JSON
    
    Args:
        completion_func: The completion function to call
        completion_kwargs: Keyword arguments to pass to completion_func
        config: Test configuration from get_memory_test_config()
        module_to_verify: Optional module/object to verify ID consistency
        module_id: Expected ID of module_to_verify
        capture_top_consumers: If True, capture top memory consumers after each batch
        top_consumers_count: Number of top consumers to capture
        output_dir: Directory to store snapshot JSON files (one file per test)
        smart_capture: If True, only capture detailed snapshots on memory increases or errors
        memory_increase_threshold_pct: Percentage increase to trigger detailed snapshot capture
        periodic_sample_interval: Capture detailed snapshot every N requests
        test_name: Name of the test (used to create snapshot filename)
        
    Returns:
        Tuple of (memory_samples, error_counts) from measurement phase
    """
    import tracemalloc as tm
    
    tm.start()
    
    try:
        # Warmup phase
        await run_warmup_phase(
            batch_size=config['batch_size'],
            warmup_batches=config['warmup_batches'],
            completion_func=completion_func,
            completion_kwargs=completion_kwargs
        )
        
        if module_to_verify is not None and module_id is not None:
            verify_module_id_consistency(module_to_verify, module_id, "after warmup")
        
        # Measurement phase
        memory_samples, error_counts = await run_measurement_phase(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            completion_func=completion_func,
            completion_kwargs=completion_kwargs,
            tracemalloc_module=tm,
            capture_top_consumers=capture_top_consumers,
            top_consumers_count=top_consumers_count,
            output_dir=output_dir,
            smart_capture=smart_capture,
            memory_increase_threshold_pct=memory_increase_threshold_pct,
            periodic_sample_interval=periodic_sample_interval,
            test_name=test_name
        )
        
        if module_to_verify is not None and module_id is not None:
            verify_module_id_consistency(module_to_verify, module_id, "after measurement")
            
    finally:
        tm.stop()
    
    return memory_samples, error_counts


def analyze_and_detect_leaks(
    memory_samples: List[float],
    error_counts: List[int],
    config: Dict[str, Any],
    module_to_verify=None,
    module_id: Optional[int] = None,
    module_name: str = "Module",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    test_name: str = "memory_test"
) -> None:
    """
    Run complete analysis phase and fail test if memory leak is detected.
    
    This high-level helper consolidates all the analysis steps:
    1. Print analysis header
    2. Check if enough samples (skip if insufficient)
    3. Check measurement noise (skip if too noisy)
    4. Prepare memory analysis (rolling average, etc.)
    5. Analyze memory growth
    6. Print growth metrics
    7. Detect memory leaks
    8. If leak detected, analyze specific leaking sources from snapshots
    9. Fail test if leak detected or print success message
    10. Print memory samples
    11. Verify module ID consistency
    
    Args:
        memory_samples: List of memory measurements from measurement phase
        error_counts: List of error counts per batch
        config: Test configuration from get_memory_test_config()
        module_to_verify: Optional module/object to verify ID consistency
        module_id: Expected ID of module_to_verify
        module_name: Display name for the module being verified
        output_dir: Directory where snapshot files are stored
        test_name: Name of the test (used to find snapshot file)
        
    Raises:
        pytest.skip: If measurements are too noisy or insufficient samples
        pytest.fail: If memory leak is detected
    """
    from ..analysis import analyze_and_report_leaking_sources
    
    print_analysis_header()
    
    if len(memory_samples) < config['sample_window'] * 2:
        pytest.skip("Not enough samples for reliable growth analysis")
    
    # Check if measurements are too noisy for reliable leak detection
    skip_on_high_noise = config.get('skip_on_high_noise', True)
    should_skip, skip_message = check_measurement_noise(memory_samples)
    if should_skip and skip_on_high_noise:
        pytest.skip(skip_message)
    elif should_skip and not skip_on_high_noise:
        print(f"WARNING: {skip_message}")
        print("Continuing test despite high noise (skip_on_high_noise=False)")
        print("Results may be unreliable due to measurement instability\n")
    
    # Calculate dynamic parameters for memory analysis
    rolling_avg, num_samples_for_avg, tail_samples = prepare_memory_analysis(
        memory_samples, config['sample_window']
    )
    
    growth_metrics = analyze_memory_growth(rolling_avg, num_samples_for_avg=num_samples_for_avg)
    print_growth_metrics(growth_metrics)
    
    # Detect memory leaks (including error-induced leaks)
    leak_detected, message = detect_memory_leak(
        growth_metrics=growth_metrics,
        memory_samples=memory_samples,
        error_counts=error_counts,
        max_growth_percent=config['max_growth_percent'],
        stabilization_tolerance_mb=config['stabilization_tolerance_mb'],
        tail_samples=tail_samples
    )
    
    if leak_detected:
        # Analyze specific leaking sources from snapshot data
        from ..constants import (
            DEFAULT_LEAK_SOURCE_FILTER_LITELLM_ONLY,
            DEFAULT_LEAK_SOURCE_MIN_GROWTH_MB,
            DEFAULT_SIGNIFICANT_MEMORY_GROWTH_PERCENT,
            DEFAULT_LEAK_SOURCE_MIN_BATCHES,
            DEFAULT_LEAK_SOURCE_MAX_RESULTS,
        )
        
        analyze_and_report_leaking_sources(
            output_dir=output_dir,
            test_name=test_name,
            filter_litellm_only=DEFAULT_LEAK_SOURCE_FILTER_LITELLM_ONLY,
            min_growth_mb=DEFAULT_LEAK_SOURCE_MIN_GROWTH_MB,
            min_growth_percent=DEFAULT_SIGNIFICANT_MEMORY_GROWTH_PERCENT,
            min_batches=DEFAULT_LEAK_SOURCE_MIN_BATCHES,
            max_results=DEFAULT_LEAK_SOURCE_MAX_RESULTS
        )
        pytest.fail(message)
    
    print(message)
    print_memory_samples(memory_samples, num_samples=10)
    
    # Final verification that ID remained consistent throughout
    if module_to_verify is not None and module_id is not None:
        verify_module_id_consistency(module_to_verify, module_id, "at test end")
        print(f"\n[TEST] ✓ {module_name} ID remained consistent throughout: {module_id}", flush=True)
    
    # Cleanup snapshot files if configured
    cleanup_snapshots = config.get('cleanup_snapshots_after_test', False)
    if cleanup_snapshots and output_dir:
        import os
        from ..snapshot.storage import sanitize_filename
        
        snapshot_filename = sanitize_filename(test_name) + ".json"
        snapshot_filepath = os.path.join(output_dir, snapshot_filename)
        
        if os.path.exists(snapshot_filepath):
            try:
                os.remove(snapshot_filepath)
                print(f"\n[CLEANUP] Snapshot file deleted: {snapshot_filepath}", flush=True)
            except Exception as e:
                print(f"\n[CLEANUP WARNING] Failed to delete snapshot file {snapshot_filepath}: {e}", flush=True)

