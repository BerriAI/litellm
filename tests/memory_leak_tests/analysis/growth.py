"""
Memory growth analysis utilities.

Provides functions for:
- Preparing memory data for analysis
- Calculating rolling averages to smooth noise
- Analyzing memory growth patterns
"""

import statistics
from typing import List, Dict, Tuple

from ..constants import DEFAULT_NUM_SAMPLES_FOR_GROWTH_ANALYSIS


def prepare_memory_analysis(
    memory_samples: List[float],
    sample_window: int
) -> Tuple[List[float], int, int]:
    """
    Calculate dynamic analysis parameters for memory leak detection.
    
    This function prepares the memory samples for analysis by:
    1. Computing a rolling average to smooth out noise
    2. Determining the number of samples to use for growth calculation
    3. Determining the number of tail samples for continuous growth detection
    
    Args:
        memory_samples: List of memory measurements in MB
        sample_window: Window size for rolling average calculation
        
    Returns:
        Tuple of:
            - rolling_avg: Smoothed memory values using rolling average
            - num_samples_for_avg: Number of samples to average at start/end for growth
            - tail_samples: Number of final samples to check for continuous growth
            
    Example:
        >>> samples = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9]
        >>> rolling_avg, num_samples, tail_samples = prepare_memory_analysis(samples, 3)
        >>> print(len(rolling_avg))
        7
    """
    # Calculate rolling average to smooth out allocator noise
    rolling_avg = calculate_rolling_average(memory_samples, sample_window)
    
    # Use 2x the sample_window for averaging (ensures we smooth over enough data)
    # Cap at 1/3 of total rolling average samples to have enough data for comparison
    num_samples_for_avg = min(sample_window * 2, len(rolling_avg) // 3)
    
    # Use 3x the sample_window for tail analysis (detect continuous growth)
    # Cap at half the total samples to ensure we're looking at a significant tail
    tail_samples = min(sample_window * 3, len(memory_samples) // 2)
    
    return rolling_avg, num_samples_for_avg, tail_samples


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
        
    Example:
        >>> samples = [10.0, 10.2, 10.1, 10.3, 10.2]
        >>> smoothed = calculate_rolling_average(samples, 2)
        >>> print(len(smoothed))
        3
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
        
    Example:
        >>> smoothed = [10.0, 10.1, 10.2, 11.0, 11.1, 11.2]
        >>> metrics = analyze_memory_growth(smoothed, num_samples_for_avg=2)
        >>> print(f"{metrics['growth_percent']:.1f}%")
        10.4%
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

