"""
Memory snapshot capture utilities.

Provides functions for:
- Capturing memory snapshots from tracemalloc
- Smart capture decision logic (when to capture detailed vs lightweight)
"""

from typing import Dict, Any, Optional

from ..constants import (
    DEFAULT_TOP_CONSUMERS_COUNT,
    DEFAULT_LIGHTWEIGHT_SNAPSHOT,
    DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT,
    DEFAULT_PERIODIC_SAMPLE_INTERVAL,
)
from ..core.cleanup import force_gc
from ..utils.conversions import bytes_to_mb


def should_capture_detailed_snapshot(
    global_request_counter: int,
    request_had_error: bool,
    current_mb: float,
    last_captured_memory_mb: float,
    memory_increase_threshold_pct: float = DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT,
    periodic_sample_interval: int = DEFAULT_PERIODIC_SAMPLE_INTERVAL
) -> bool:
    """
    Determine if a detailed memory snapshot should be captured for this request.
    
    Smart capture strategy captures detailed data only when:
    1. First request (baseline)
    2. Error occurred
    3. Memory increased by more than threshold percentage
    4. Periodic sampling (every N requests)
    
    Args:
        global_request_counter: Global counter for all requests
        request_had_error: Whether this request encountered an error
        current_mb: Current memory usage in MB
        last_captured_memory_mb: Memory usage at last detailed capture in MB
        memory_increase_threshold_pct: Percentage increase to trigger capture
        periodic_sample_interval: Capture every N requests for periodic sampling
        
    Returns:
        True if detailed snapshot should be captured, False for lightweight
        
    Example:
        >>> should_capture = should_capture_detailed_snapshot(
        ...     global_request_counter=1,
        ...     request_had_error=False,
        ...     current_mb=10.5,
        ...     last_captured_memory_mb=0.0
        ... )
        >>> print(should_capture)  # True - first request
        True
    """
    # Always capture first request as baseline
    if global_request_counter == 1:
        return True
    
    # Always capture on errors
    if request_had_error:
        return True
    
    # Capture on periodic interval
    if global_request_counter % periodic_sample_interval == 0:
        return True
    
    # Capture on significant memory increase
    if last_captured_memory_mb > 0:
        memory_increase_pct = ((current_mb - last_captured_memory_mb) / last_captured_memory_mb) * 100
        if memory_increase_pct > memory_increase_threshold_pct:
            return True
    
    return False


def capture_request_memory_snapshot(
    tracemalloc_module,
    global_request_counter: int,
    batch: int,
    req_idx: int,
    request_had_error: bool,
    error_message: Optional[str],
    top_consumers_count: int = DEFAULT_TOP_CONSUMERS_COUNT,
    lightweight: bool = DEFAULT_LIGHTWEIGHT_SNAPSHOT
) -> Dict[str, Any]:
    """
    Capture memory snapshot after a request and return the data.
    
    This function:
    1. Forces garbage collection
    2. Captures current and peak memory using tracemalloc
    3. Takes a snapshot of memory allocations (unless lightweight=True)
    4. Extracts top memory consumers (unless lightweight=True)
    5. Creates a structured request data dictionary
    6. Returns the data (caller is responsible for saving)
    
    Args:
        tracemalloc_module: The tracemalloc module for memory tracking
        global_request_counter: Global counter for all requests
        batch: Current batch index (0-indexed)
        req_idx: Request index within the batch (0-indexed)
        request_had_error: Whether this request encountered an error
        error_message: Error message if request_had_error is True
        top_consumers_count: Number of top memory consumers to capture
        lightweight: If True, skip expensive snapshot operations (top consumers)
        
    Returns:
        Dictionary containing memory snapshot data for this request
        
    Example:
        >>> import tracemalloc
        >>> tracemalloc.start()
        >>> snapshot_data = capture_request_memory_snapshot(
        ...     tracemalloc_module=tracemalloc,
        ...     global_request_counter=1,
        ...     batch=0,
        ...     req_idx=0,
        ...     request_had_error=False,
        ...     error_message=None
        ... )
        >>> print(snapshot_data['request_id'])
        'req-1'
    """
    import datetime
    
    force_gc()
    current, peak = tracemalloc_module.get_traced_memory()
    current_mb = bytes_to_mb(current)
    peak_mb = bytes_to_mb(peak)
    
    request_data = {
        'request_id': f'req-{global_request_counter}',
        'batch': batch + 1,
        'request_in_batch': req_idx + 1,
        'timestamp': datetime.datetime.now().isoformat(),
        'current_mb': round(current_mb, 3),
        'peak_mb': round(peak_mb, 3),
        'had_error': request_had_error,
        'error_message': error_message,
    }
    
    # Only capture expensive top consumers data if not in lightweight mode
    if not lightweight:
        snapshot = tracemalloc_module.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        request_data['top_consumers'] = [
            {
                'file': str(stat.traceback.format()[0]) if stat.traceback else 'unknown',
                'size_mb': round(bytes_to_mb(stat.size), 3),
                'count': stat.count
            }
            for stat in top_stats[:top_consumers_count]
        ]
    
    return request_data

