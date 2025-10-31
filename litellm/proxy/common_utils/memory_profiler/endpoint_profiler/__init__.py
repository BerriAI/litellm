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

# Main profiler classes
from .profiler import EndpointProfiler, profile_endpoint

# Storage utilities
from .storage import ProfileBuffer

# Analysis utilities
from .analyze_profiles import (
    analyze_endpoint_memory,
    analyze_profile_file,
    load_profile_data,
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
    # Main classes
    "EndpointProfiler",
    "profile_endpoint",
    "ProfileBuffer",
    
    # Analysis functions
    "analyze_endpoint_memory",
    "analyze_profile_file",
    "load_profile_data",
    
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
