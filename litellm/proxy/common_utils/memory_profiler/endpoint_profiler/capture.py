"""
Memory snapshot capture utilities for endpoint profiling.

Provides functions for:
- Capturing memory snapshots using tracemalloc
- Smart capture decision logic (when to capture detailed vs lightweight)
- Request profile data capture

Leverages common functionality from parent memory_profiler.snapshot for consistency.
"""

import linecache
import tracemalloc
from typing import Any, Dict, Optional

from .constants import (
    DEFAULT_CAPTURE_DETAILED_ON_ERROR,
    MEMORY_DECIMAL_PLACES,
    TIMING_DECIMAL_PLACES,
)
from .utils import bytes_to_mb, force_gc, get_response_status_category, is_error_response

# Import smart capture logic and constants from parent memory_profiler module
from ..snapshot.capture import should_capture_detailed_snapshot as _should_capture_detailed_snapshot
from ..constants import (
    DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT,
    DEFAULT_PERIODIC_SAMPLE_INTERVAL,
    DEFAULT_TOP_CONSUMERS_COUNT,
)


def should_capture_detailed_snapshot(
    request_counter: int,
    request_had_error: bool,
    current_memory_mb: float,
    last_captured_memory_mb: float,
    memory_increase_threshold_pct: float = DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT,
    periodic_sample_interval: int = DEFAULT_PERIODIC_SAMPLE_INTERVAL,
    capture_detailed_on_error: bool = DEFAULT_CAPTURE_DETAILED_ON_ERROR,
) -> bool:
    """
    Determine if a detailed snapshot should be captured for this request.
    
    Wrapper around parent module's implementation with endpoint-specific parameters.
    
    Smart capture strategy captures detailed data only when:
    1. First request (baseline)
    2. Error occurred (4xx or 5xx)
    3. Memory increased by more than threshold percentage
    4. Periodic sampling (every N requests)
    
    Args:
        request_counter: Global counter for all requests
        request_had_error: Whether this request had an error response
        current_memory_mb: Current memory usage in MB
        last_captured_memory_mb: Memory usage at last detailed capture in MB
        memory_increase_threshold_pct: Percentage increase to trigger capture
        periodic_sample_interval: Capture every N requests for periodic sampling
        capture_detailed_on_error: Whether to capture detailed on errors
        
    Returns:
        True if detailed snapshot should be captured, False for lightweight
        
    Example:
        >>> should_capture_detailed_snapshot(
        ...     request_counter=1,
        ...     request_had_error=False,
        ...     current_memory_mb=100.0,
        ...     last_captured_memory_mb=0.0
        ... )
        True  # First request
    """
    # Use parent module's implementation
    return _should_capture_detailed_snapshot(
        global_request_counter=request_counter,
        request_had_error=request_had_error,
        current_mb=current_memory_mb,
        last_captured_memory_mb=last_captured_memory_mb,
        memory_increase_threshold_pct=memory_increase_threshold_pct,
        periodic_sample_interval=periodic_sample_interval,
    )


def capture_memory_snapshot(
    top_consumers_count: int = DEFAULT_TOP_CONSUMERS_COUNT,
    lightweight: bool = False,
) -> Dict[str, Any]:
    """
    Capture current memory snapshot using tracemalloc.
    
    Args:
        top_consumers_count: Number of top memory consumers to capture
        lightweight: If True, skip expensive snapshot operations
        
    Returns:
        Dictionary containing memory data:
        - current_mb: Current memory usage
        - peak_mb: Peak memory usage
        - top_consumers: List of top memory consuming files (if not lightweight)
        - tracking_enabled: Whether tracemalloc is enabled
        
    Example:
        >>> snapshot = capture_memory_snapshot(top_consumers_count=10)
        >>> print(snapshot['current_mb'])
        125.4
    """
    if not tracemalloc.is_tracing():
        return {
            'current_mb': 0.0,
            'peak_mb': 0.0,
            'tracking_enabled': False,
        }
    
    # Only call force_gc() on detailed snapshots to reduce overhead
    # Lightweight snapshots skip GC for better performance
    if not lightweight:
        force_gc()
    
    current, peak = tracemalloc.get_traced_memory()
    current_mb = bytes_to_mb(current)
    peak_mb = bytes_to_mb(peak)
    
    result = {
        'current_mb': round(current_mb, MEMORY_DECIMAL_PLACES),
        'peak_mb': round(peak_mb, MEMORY_DECIMAL_PLACES),
        'tracking_enabled': True,
    }
    
    # Only capture expensive top consumers data if not in lightweight mode
    if not lightweight:
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            result['top_consumers'] = [
                {
                    # Directly access filename and lineno (avoid format() overhead)
                    'file': f"{stat.traceback[0].filename}:{stat.traceback[0].lineno}" if stat.traceback else 'unknown',
                    'size_mb': round(bytes_to_mb(stat.size), MEMORY_DECIMAL_PLACES),
                    'count': stat.count,
                }
                for stat in top_stats[:top_consumers_count]
            ]
            
            # CRITICAL: Clear linecache after capturing traceback info
            # Accessing traceback.filename triggers linecache to cache source files.
            # We must clear this cache to prevent unbounded memory growth.
            linecache.clearcache()
            
        except Exception as e:
            result['top_consumers_error'] = str(e)
    
    return result


def capture_request_profile(
    endpoint: str,
    method: str,
    request_counter: int,
    start_time: float,
    end_time: float,
    status_code: int,
    error_message: Optional[str] = None,
    _request: Optional[Any] = None,
    _response: Optional[Any] = None,
    last_detailed_memory_mb: float = 0.0,
    track_memory: bool = True,
    lightweight: bool = False,
) -> Dict[str, Any]:
    """
    Capture complete profile data for a request.
    
    This function:
    1. Calculates request latency
    2. Determines error status
    3. Captures memory snapshot (if enabled)
    4. Uses smart capture to decide detail level
    5. Returns structured profile dictionary
    
    Args:
        endpoint: Endpoint path (e.g., "/chat/completions")
        method: HTTP method (e.g., "POST")
        request_counter: Global request counter
        start_time: Request start timestamp (from time.time())
        end_time: Request end timestamp (from time.time())
        status_code: HTTP response status code
        error_message: Error message if request failed
        _request: FastAPI Request object (optional, unused - kept for compatibility)
        _response: Response object or data (optional, unused - kept for compatibility)
        last_detailed_memory_mb: Memory at last detailed capture
        track_memory: Whether to track memory usage
        lightweight: Whether to use lightweight capture mode
        
    Returns:
        Dictionary containing complete profile data
        
    Example:
        >>> import time
        >>> start = time.time()
        >>> # ... process request ...
        >>> end = time.time()
        >>> profile = capture_request_profile(
        ...     endpoint="/chat/completions",
        ...     method="POST",
        ...     request_counter=1,
        ...     start_time=start,
        ...     end_time=end,
        ...     status_code=200
        ... )
        >>> print(profile['latency'])
        0.123456
    """
    latency = end_time - start_time
    had_error = is_error_response(status_code)
    
    # Build base profile
    # Use end_time directly as float timestamp for performance (avoid datetime overhead)
    profile = {
        'request_id': f'req-{request_counter}',
        'timestamp': end_time,  # Float timestamp is faster than datetime.isoformat()
        'endpoint': endpoint,
        'method': method,
        'status_code': status_code,
        'status_category': get_response_status_category(status_code),
        'latency': round(latency, TIMING_DECIMAL_PLACES),
        'had_error': had_error,
    }
    
    if error_message:
        profile['error_message'] = error_message
    
    # Capture memory if enabled
    if track_memory:
        memory_data = capture_memory_snapshot(lightweight=lightweight)
        current_memory_mb = memory_data.get('current_mb', 0.0)
        
        # Determine if we should capture detailed snapshot
        should_detailed = should_capture_detailed_snapshot(
            request_counter=request_counter,
            request_had_error=had_error,
            current_memory_mb=current_memory_mb,
            last_captured_memory_mb=last_detailed_memory_mb,
        )
        
        # If lightweight was forced but we should capture detailed, recapture
        if lightweight and should_detailed:
            memory_data = capture_memory_snapshot(lightweight=False)
        
        profile['memory'] = memory_data
        profile['detailed_snapshot'] = should_detailed
    
    return profile
