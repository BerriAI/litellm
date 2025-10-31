"""
Memory leak profiling utilities for LiteLLM Proxy.

This package provides lean, memory-focused profiling and leak detection with:
- Memory tracking using tracemalloc
- Professional memory leak detection (leveraging parent memory_profiler module)
- Smart capture (detailed vs lightweight snapshots)
- Efficient buffering (not writing on every request)
- Automatic rotation to prevent unbounded file growth

Key Features:
- Reuses common functionality from parent memory_profiler module
- Focused exclusively on memory profiling
- Professional leak detection algorithms
- Minimal overhead with smart sampling

Usage:
    >>> from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import (
    ...     EndpointProfiler,
    ...     profile_endpoint,
    ... )
    >>> 
    >>> # Initialize profiler (typically done once at server startup)
    >>> profiler = EndpointProfiler.get_instance(
    ...     enabled=True,
    ...     sampling_rate=1.0,  # Profile all requests
    ...     buffer_size=100,     # Flush every 100 requests
    ... )
    >>> 
    >>> # Start auto-flush background task
    >>> await profiler.start_auto_flush()
    >>> 
    >>> # Use decorator on endpoints
    >>> @profile_endpoint()
    >>> async def chat_completions(request: Request):
    ...     return {"response": "data"}
    >>> 
    >>> # Get stats
    >>> stats = profiler.get_stats()
    >>> 
    >>> # Manual flush
    >>> profiler.buffer.flush_all()

Analysis:
    >>> from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import (
    ...     analyze_profile_file,
    ...     load_profile_data,
    ...     analyze_endpoint_memory,
    ... )
    >>> 
    >>> # Analyze a profile file
    >>> analysis = analyze_profile_file("endpoint_profiles/chat_completions.json")
    >>> 
    >>> # Check for leaks
    >>> if analysis.get('leak_detected'):
    ...     print(f"Leak detected: {analysis['leak_message']}")
"""

# Main profiler classes (modular implementation)
# profiler.py re-exports from profiler_core.py and decorator.py
from .profiler import EndpointProfiler, profile_endpoint

# Core profiler components (can be imported directly if needed)
from .profiler_core import EndpointProfiler as _EndpointProfilerCore
from .decorator import profile_endpoint as _profile_endpoint_decorator

# Storage utilities
from .storage import ProfileBuffer

# Analysis utilities (main orchestration)
from .analyze_profiles import (
    analyze_profile_file,
)

# Data loading utilities
from .data_loading import (
    load_profile_data,
    extract_request_number,
    sort_profiles_by_request_id,
)

# Memory analysis utilities
from .memory_analysis import (
    analyze_endpoint_memory,
    extract_memory_samples,
)

# Consumer analysis utilities
from .consumer_analysis import (
    analyze_top_memory_consumers,
    aggregate_memory_by_location,
    get_top_memory_consumers,
    parse_file_location,
)

# Location analysis utilities
from .location_analysis import (
    analyze_memory_growth_by_location,
    track_location_memory_over_time,
    calculate_location_growth,
    get_top_growing_locations,
)

# Reporting utilities
from .reporting import (
    print_analysis_report,
    print_loading_info,
    print_error,
)

# Capture utilities (for advanced usage)
from .capture import (
    capture_memory_snapshot,
    capture_request_profile,
    should_capture_detailed_snapshot,
)

# Utility functions (for advanced usage)
from .utils import (
    bytes_to_mb,
    force_gc,
    format_latency,
    format_memory,
    sanitize_endpoint_name,
)

# Constants (for configuration)
# Import shared constants from parent module
from ..constants import (
    DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT,
    DEFAULT_PERIODIC_SAMPLE_INTERVAL,
)
from .constants import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_FLUSH_INTERVAL_SECONDS,
    DEFAULT_MAX_PROFILES_PER_ENDPOINT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROFILING_ENABLED,
    DEFAULT_SAMPLING_RATE,
)

__all__ = [
    # Main classes (from modular implementation)
    "EndpointProfiler",
    "profile_endpoint",
    "ProfileBuffer",
    
    # Analysis functions (main orchestration)
    "analyze_profile_file",
    
    # Data loading
    "load_profile_data",
    "extract_request_number",
    "sort_profiles_by_request_id",
    
    # Memory analysis
    "analyze_endpoint_memory",
    "extract_memory_samples",
    
    # Consumer analysis
    "analyze_top_memory_consumers",
    "aggregate_memory_by_location",
    "get_top_memory_consumers",
    "parse_file_location",
    
    # Location analysis
    "analyze_memory_growth_by_location",
    "track_location_memory_over_time",
    "calculate_location_growth",
    "get_top_growing_locations",
    
    # Reporting
    "print_analysis_report",
    "print_loading_info",
    "print_error",
    
    # Capture functions (advanced)
    "capture_memory_snapshot",
    "capture_request_profile",
    "should_capture_detailed_snapshot",
    
    # Utility functions (advanced)
    "bytes_to_mb",
    "force_gc",
    "format_latency",
    "format_memory",
    "sanitize_endpoint_name",
    
    # Constants (configuration)
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_FLUSH_INTERVAL_SECONDS",
    "DEFAULT_MAX_PROFILES_PER_ENDPOINT",
    "DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PERIODIC_SAMPLE_INTERVAL",
    "DEFAULT_PROFILING_ENABLED",
    "DEFAULT_SAMPLING_RATE",
]
