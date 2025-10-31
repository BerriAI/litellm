"""
Unit conversion utilities for memory leak testing.

Provides functions to convert between different units of measurement
used in memory profiling and analysis.
"""


def bytes_to_mb(bytes_value: int) -> float:
    """
    Convert bytes to megabytes.
    
    Args:
        bytes_value: Value in bytes
        
    Returns:
        Value in megabytes (MB)
        
    Example:
        >>> bytes_to_mb(1048576)
        1.0
        >>> bytes_to_mb(5242880)
        5.0
    """
    return bytes_value / (1024 * 1024)

