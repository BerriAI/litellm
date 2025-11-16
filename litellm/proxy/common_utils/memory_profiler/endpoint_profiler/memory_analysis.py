"""
Core memory analysis for endpoint profiling.

This module provides functions for analyzing memory usage patterns in endpoint profiles.

Features:
- Extract memory samples from profiles
- Detect memory leaks using parent module's algorithms
- Prepare data for analysis
"""

from typing import Any, Dict, List, Tuple

# Import memory leak detection from parent memory_profiler module
from ..analysis.detection import detect_memory_leak
from ..analysis.growth import (
    analyze_memory_growth,
    prepare_memory_analysis,
)
from ..constants import DEFAULT_ROLLING_AVERAGE_WINDOW

from .data_loading import sort_profiles_by_request_id


def extract_memory_samples(profiles: List[Dict[str, Any]]) -> Tuple[List[float], List[int]]:
    """
    Extract memory samples and error counts from profiles.
    
    Args:
        profiles: List of profile dictionaries
        
    Returns:
        Tuple of (memory_samples, error_counts) - one per profile
        
    Example:
        >>> profiles = [{"memory": {"current_mb": 100.5}, "had_error": False}]
        >>> memory_samples, error_counts = extract_memory_samples(profiles)
        >>> memory_samples[0]
        100.5
        >>> error_counts[0]
        0
    """
    memory_samples = []
    error_counts = []
    
    sorted_profiles = sort_profiles_by_request_id(profiles)
    
    for profile in sorted_profiles:
        if 'memory' in profile and 'current_mb' in profile['memory']:
            memory_samples.append(profile['memory']['current_mb'])
            error_counts.append(1 if profile.get('had_error', False) else 0)
    
    return memory_samples, error_counts


def analyze_endpoint_memory(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze memory growth and detect leaks for an endpoint.
    
    Uses algorithms from parent memory_profiler module for leak detection.
    
    Args:
        profiles: List of profile dictionaries
        
    Returns:
        Dictionary with analysis results including:
        - total_requests: Number of requests analyzed
        - memory_samples: Raw memory samples
        - error_counts: Error counts per request
        - rolling_average: Smoothed memory values
        - growth_metrics: Memory growth analysis
        - leak_detected: Whether a leak was detected
        - leak_message: Description of leak status
        - error: Error message if analysis failed
        
    Example:
        >>> profiles = load_profile_data("endpoint_profiles/chat_completions.json")
        >>> analysis = analyze_endpoint_memory(profiles)
        >>> if analysis.get('leak_detected'):
        ...     print(f"Leak: {analysis['leak_message']}")
    """
    if not profiles:
        return {
            'error': 'No profiles provided'
        }
    
    memory_samples, error_counts = extract_memory_samples(profiles)
    
    if not memory_samples:
        return {
            'error': 'No memory data found in profiles'
        }
    
    try:
        # Prepare memory data
        rolling_avg, num_samples, tail_samples = prepare_memory_analysis(
            memory_samples,
            sample_window=DEFAULT_ROLLING_AVERAGE_WINDOW
        )
        
        # Analyze memory growth
        growth_metrics = analyze_memory_growth(rolling_avg, num_samples)
        
        # Detect memory leaks
        leak_detected, leak_message = detect_memory_leak(
            growth_metrics=growth_metrics,
            memory_samples=memory_samples,
            error_counts=error_counts,
            tail_samples=tail_samples
        )
        
        return {
            'total_requests': len(memory_samples),
            'memory_samples': memory_samples,
            'error_counts': error_counts,
            'rolling_average': rolling_avg,
            'growth_metrics': growth_metrics,
            'leak_detected': leak_detected,
            'leak_message': leak_message,
        }
        
    except Exception as e:
        return {
            'error': f'Error during analysis: {str(e)}'
        }

