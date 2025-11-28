"""
Constants and configuration values for endpoint memory profiling.

This module provides centralized constants for profiling memory usage of proxy endpoints.
Imports shared constants from parent memory_profiler module for consistency.

Constants are organized into sections:
1. Profiling Configuration - Buffer sizes, flush intervals (endpoint-specific)
2. Smart Capture Configuration - Imported from parent module
3. Storage Configuration - File paths and rotation settings (endpoint-specific)
4. Profiler State - Runtime control (endpoint-specific)
"""

import os


# =============================================================================
# Profiling Configuration (Endpoint-Specific)
# =============================================================================

# Number of requests to buffer in memory before flushing to disk
# Optimized for production: flush every 100 requests
DEFAULT_BUFFER_SIZE = 100

# Time in seconds to force flush buffer even if not full
# Prevents data loss during low traffic periods
DEFAULT_FLUSH_INTERVAL_SECONDS = 60

# Maximum number of profile entries to keep per endpoint
# Older entries are rotated out to prevent unbounded file growth
DEFAULT_MAX_PROFILES_PER_ENDPOINT = 10000

# Whether to enable async flushing (non-blocking writes)
DEFAULT_ASYNC_FLUSH = True

# =============================================================================
# Smart Capture Configuration  
# =============================================================================
# These are imported from parent module above:
# - DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT
# - DEFAULT_PERIODIC_SAMPLE_INTERVAL  
# - DEFAULT_TOP_CONSUMERS_COUNT

# Whether to capture detailed snapshots on error responses (4xx, 5xx)
DEFAULT_CAPTURE_DETAILED_ON_ERROR = True

# =============================================================================
# Storage Configuration
# =============================================================================

# Default directory for profile data files
DEFAULT_OUTPUT_DIR = os.getenv(
    "LITELLM_PROFILE_OUTPUT_DIR",
    "endpoint_profiles"
)

# Whether to delete profile files on server shutdown
DEFAULT_CLEANUP_ON_SHUTDOWN = False

# File format for profile output (json only for now)
DEFAULT_FILE_FORMAT = "json"

# =============================================================================
# Profiler State
# =============================================================================

# Whether profiling is enabled globally
DEFAULT_PROFILING_ENABLED = os.getenv(
    "LITELLM_ENABLE_PROFILING", 
    "true"
).lower() in ("true", "1", "yes")

# Sampling rate for profiling (0.0 to 1.0)
# 1.0 = profile all requests, 0.1 = profile 10% of requests
DEFAULT_SAMPLING_RATE = float(os.getenv("LITELLM_PROFILING_SAMPLE_RATE", "1.0"))

# =============================================================================
# Format Configuration
# =============================================================================

# Number of decimal places for timing metrics (seconds)
TIMING_DECIMAL_PLACES = 6

# Number of decimal places for memory metrics (MB)
MEMORY_DECIMAL_PLACES = 3

