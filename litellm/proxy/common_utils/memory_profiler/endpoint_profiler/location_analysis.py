"""
Memory growth by location analysis for endpoint profiling.

This module provides functions for analyzing how memory usage at specific
code locations grows over time.

Features:
- Track first and last appearance of each location
- Calculate growth metrics (absolute and percentage)
- Identify locations with significant memory growth
"""

from typing import Any, Dict, List

from .consumer_analysis import parse_file_location, SEPARATOR_WIDTH, DEFAULT_TOP_N
from .data_loading import sort_profiles_by_request_id


def track_location_memory_over_time(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Track memory usage for each location over time.
    
    Records first and last appearance, along with all samples.
    
    Args:
        profiles: List of profile dictionaries
        
    Returns:
        Dictionary mapping location to temporal data:
        - first_mb: Memory at first appearance
        - first_idx: Index of first appearance
        - last_mb: Memory at last appearance
        - last_idx: Index of last appearance
        - samples: List of all memory samples
        
    Example:
        >>> profiles = [{"memory": {"top_consumers": [{"file": "test.py:1", "size_mb": 10.0}]}}]
        >>> tracking = track_location_memory_over_time(profiles)
        >>> tracking["test.py:1"]["first_mb"]
        10.0
    """
    sorted_profiles = sort_profiles_by_request_id(profiles)
    location_data: Dict[str, Dict[str, Any]] = {}
    
    for idx, profile in enumerate(sorted_profiles):
        if 'memory' not in profile or 'top_consumers' not in profile['memory']:
            continue
        
        for consumer in profile['memory']['top_consumers']:
            file_info = consumer['file'].strip()
            size_mb = consumer['size_mb']
            
            location = parse_file_location(file_info)
            
            if location not in location_data:
                location_data[location] = {
                    'first_mb': size_mb,
                    'first_idx': idx,
                    'last_mb': size_mb,
                    'last_idx': idx,
                    'samples': [size_mb]
                }
            else:
                location_data[location]['last_mb'] = size_mb
                location_data[location]['last_idx'] = idx
                location_data[location]['samples'].append(size_mb)
    
    return location_data


def calculate_location_growth(location_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate memory growth metrics for each location.
    
    Args:
        location_data: Temporal memory data from track_location_memory_over_time()
        
    Returns:
        List of dictionaries with growth metrics:
        - location: File:line location
        - first_mb: Initial memory
        - last_mb: Final memory
        - min_mb: Minimum memory
        - max_mb: Maximum memory
        - growth_mb: Absolute growth
        - growth_percent: Percentage growth
        - sample_count: Number of samples
        
    Example:
        >>> location_data = {"test.py:1": {"first_mb": 10.0, "last_mb": 20.0, "samples": [10.0, 20.0]}}
        >>> growth = calculate_location_growth(location_data)
        >>> growth[0]["growth_mb"]
        10.0
    """
    growth_data = []
    
    for location, data in location_data.items():
        # Skip locations that only appeared once
        if data['first_idx'] == data['last_idx']:
            continue
        
        growth_mb = data['last_mb'] - data['first_mb']
        growth_percent = (growth_mb / data['first_mb'] * 100) if data['first_mb'] > 0 else 0
        
        min_mb = min(data['samples'])
        max_mb = max(data['samples'])
        
        growth_data.append({
            'location': location,
            'first_mb': data['first_mb'],
            'last_mb': data['last_mb'],
            'min_mb': min_mb,
            'max_mb': max_mb,
            'growth_mb': growth_mb,
            'growth_percent': growth_percent,
            'sample_count': len(data['samples']),
        })
    
    return growth_data


def get_top_growing_locations(growth_data: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N) -> List[Dict[str, Any]]:
    """
    Get top N locations by absolute memory growth.
    
    Args:
        growth_data: List of growth metrics from calculate_location_growth()
        top_n: Number of top locations to return
        
    Returns:
        List of growth data dictionaries sorted by absolute growth (descending)
        
    Example:
        >>> growth_data = [{"growth_mb": 100.0}, {"growth_mb": 50.0}]
        >>> top = get_top_growing_locations(growth_data, top_n=1)
        >>> top[0]["growth_mb"]
        100.0
    """
    return sorted(growth_data, key=lambda x: abs(x['growth_mb']), reverse=True)[:top_n]


def analyze_memory_growth_by_location(profiles: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N) -> None:
    """
    Analyze memory growth by location (first vs last appearance).
    
    Shows which files/lines are growing in memory over time.
    Prints formatted report with top growing locations.
    
    Args:
        profiles: List of profile dictionaries
        top_n: Number of top growing locations to show
        
    Example:
        >>> profiles = load_profile_data("endpoint_profiles/chat_completions.json")
        >>> analyze_memory_growth_by_location(profiles, top_n=10)
    """
    # Track memory over time
    location_data = track_location_memory_over_time(profiles)
    
    # Calculate growth
    growth_data = calculate_location_growth(location_data)
    
    # Get top growing
    sorted_growth = get_top_growing_locations(growth_data, top_n)
    
    if not sorted_growth:
        print("\n" + "="*SEPARATOR_WIDTH)  # noqa: T201
        print("MEMORY GROWTH BY LOCATION")  # noqa: T201
        print("="*SEPARATOR_WIDTH)  # noqa: T201
        print("No locations with multiple samples found for analysis.")  # noqa: T201
        print("="*SEPARATOR_WIDTH + "\n")  # noqa: T201
        return
    
    print("\n" + "="*SEPARATOR_WIDTH)  # noqa: T201
    print(f"TOP {top_n} LOCATIONS BY MEMORY GROWTH (First → Last)  # noqa: T201")
    print("="*SEPARATOR_WIDTH)  # noqa: T201
    print(f"{'Rank':<6} {'Growth (MB)  # noqa: T201':<14} {'Growth %':<12} {'Range (MB)':<20} {'Samples':<10} {'Location'}")
    print("-"*SEPARATOR_WIDTH)  # noqa: T201
    
    for rank, data in enumerate(sorted_growth, 1):
        range_str = f"{data['min_mb']:.2f} → {data['max_mb']:.2f}"
        location = data['location']
        
        print(f"{rank:<6} {data['growth_mb']:<14.3f} {data['growth_percent']:<12.1f} {range_str:<20} {data['sample_count']:<10} {location}")  # noqa: T201
    
    print("="*SEPARATOR_WIDTH + "\n")  # noqa: T201

