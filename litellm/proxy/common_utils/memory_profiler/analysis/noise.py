"""
Noise detection for memory measurements.

Provides functions to assess measurement stability and determine
if the test environment is too unstable for reliable leak detection.
"""

import statistics
from typing import List, Tuple

from ..constants import DEFAULT_MAX_COEFFICIENT_VARIATION


def check_measurement_noise(
    memory_samples: List[float],
    max_coefficient_variation: float = DEFAULT_MAX_COEFFICIENT_VARIATION
) -> Tuple[bool, str]:
    """
    Check if memory measurements are too noisy for reliable leak detection.
    
    Calculates the coefficient of variation (CV) to assess measurement stability.
    High CV indicates the test environment is unstable and results would be unreliable.
    
    Args:
        memory_samples: List of memory measurements in MB
        max_coefficient_variation: Maximum acceptable CV percentage (default: 40%)
        
    Returns:
        Tuple of (should_skip: bool, skip_message: str)
        If should_skip is True, the test should be skipped with the provided message.
        
    Example:
        >>> samples = [10.1, 10.2, 10.3, 10.2, 10.4]
        >>> should_skip, message = check_measurement_noise(samples)
        >>> print(should_skip)
        False
    """
    if len(memory_samples) <= 1:
        return False, ""
    
    memory_std_dev = statistics.stdev(memory_samples)
    memory_mean = statistics.mean(memory_samples)
    memory_coefficient_variation = (memory_std_dev / memory_mean * 100) if memory_mean > 0 else 0
    
    print(f"Memory std dev: {memory_std_dev:.3f} MB ({memory_coefficient_variation:.1f}% coefficient of variation)")
    
    if memory_coefficient_variation > max_coefficient_variation:
        skip_message = f"Memory measurements too noisy (CV={memory_coefficient_variation:.1f}%) - test environment unstable"
        return True, skip_message
    
    return False, ""

