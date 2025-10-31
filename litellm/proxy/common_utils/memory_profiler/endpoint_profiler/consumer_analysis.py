"""
Top memory consumers analysis for endpoint profiling.

This module provides functions for analyzing which code locations consume the most memory.

Features:
- Aggregate memory usage by file:line location
- Find top memory consuming locations
- Calculate statistics (total, average, max)
"""

from collections import defaultdict
from typing import Any, Dict, List


# Display/reporting parameters
DEFAULT_TOP_N = 20  # Default number of top items to display
SEPARATOR_WIDTH = 80  # Width of separator lines in reports


def parse_file_location(file_info: str) -> str:
    """
    Parse and clean up file location string.
    
    Converts tracemalloc format to clean file:line format.
    
    Args:
        file_info: Raw file info from tracemalloc (e.g., 'File "path.py", line 123')
        
    Returns:
        Clean location string (e.g., "path.py:123")
        
    Example:
        >>> parse_file_location('File "/path/to/file.py", line 42')
        '/path/to/file.py:42'
    """
    location = file_info.strip()
    
    # Parse tracemalloc format: File "path", line N
    if 'File' in file_info and 'line' in file_info:
        parts = file_info.split('"')
        if len(parts) >= 2:
            file_path = parts[1]
            line_parts = file_info.split('line ')
            line_num = line_parts[1].split(',')[0] if len(line_parts) > 1 else '?'
            location = f"{file_path}:{line_num}"
    
    return location


def aggregate_memory_by_location(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate memory usage by file:line location across all profiles.
    
    Args:
        profiles: List of profile dictionaries with memory data
        
    Returns:
        Dictionary mapping location to aggregated statistics:
        - total_mb: Total memory across all samples
        - max_mb: Maximum memory seen
        - count: Number of samples
        - samples: List of individual sample values
        
    Example:
        >>> profiles = [{"memory": {"top_consumers": [{"file": "test.py:1", "size_mb": 10.0}]}}]
        >>> aggregated = aggregate_memory_by_location(profiles)
        >>> aggregated["test.py:1"]["total_mb"]
        10.0
    """
    memory_by_location: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'total_mb': 0.0,
        'max_mb': 0.0,
        'count': 0,
        'samples': []
    })
    
    for profile in profiles:
        if 'memory' not in profile or 'top_consumers' not in profile['memory']:
            continue
        
        for consumer in profile['memory']['top_consumers']:
            file_info = consumer['file'].strip()
            size_mb = consumer['size_mb']
            
            location = parse_file_location(file_info)
            
            memory_by_location[location]['total_mb'] += size_mb
            memory_by_location[location]['max_mb'] = max(
                memory_by_location[location]['max_mb'], size_mb
            )
            memory_by_location[location]['count'] += 1
            memory_by_location[location]['samples'].append(size_mb)
    
    return memory_by_location


def get_top_memory_consumers(
    memory_by_location: Dict[str, Dict[str, Any]], 
    top_n: int = DEFAULT_TOP_N
) -> List[tuple[str, Dict[str, Any]]]:
    """
    Get top N locations by total memory usage.
    
    Args:
        memory_by_location: Aggregated memory data by location
        top_n: Number of top consumers to return
        
    Returns:
        List of (location, data) tuples sorted by total memory (descending)
        
    Example:
        >>> memory_data = {"file1.py:1": {"total_mb": 100.0}, "file2.py:1": {"total_mb": 50.0}}
        >>> top = get_top_memory_consumers(memory_data, top_n=1)
        >>> top[0][0]
        'file1.py:1'
    """
    return sorted(
        memory_by_location.items(),
        key=lambda x: x[1]['total_mb'],
        reverse=True
    )[:top_n]


def analyze_top_memory_consumers(profiles: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N) -> None:
    """
    Aggregate and analyze top memory consumers across all requests.
    
    Prints formatted report showing top memory-consuming locations.
    
    Args:
        profiles: List of profile dictionaries
        top_n: Number of top consumers to show
        
    Example:
        >>> profiles = load_profile_data("endpoint_profiles/chat_completions.json")
        >>> analyze_top_memory_consumers(profiles, top_n=10)
    """
    memory_by_location = aggregate_memory_by_location(profiles)
    sorted_locations = get_top_memory_consumers(memory_by_location, top_n)
    
    print("\n" + "="*SEPARATOR_WIDTH)
    print(f"TOP {top_n} MEMORY-CONSUMING LOCATIONS (Aggregated)")
    print("="*SEPARATOR_WIDTH)
    print(f"{'Rank':<6} {'Total (MB)':<12} {'Max (MB)':<12} {'Avg (MB)':<12} {'Samples':<10} {'Location'}")
    print("-"*SEPARATOR_WIDTH)
    
    for rank, (location, data) in enumerate(sorted_locations, 1):
        avg_mb = data['total_mb'] / data['count']
        print(f"{rank:<6} {data['total_mb']:<12.3f} {data['max_mb']:<12.3f} {avg_mb:<12.3f} {data['count']:<10} {location}")
    
    print("="*SEPARATOR_WIDTH + "\n")

