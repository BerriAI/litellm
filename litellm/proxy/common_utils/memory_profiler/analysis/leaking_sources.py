"""
Memory leak source identification - analyzes top consumers across batches.

Provides functions for:
- Parsing file paths and line numbers from tracemalloc output
- Aggregating memory usage by file:line across batches
- Identifying which files/lines are leaking memory
- Reporting memory growth for specific sources
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from ..snapshot.storage import sanitize_filename
from ..constants import (
    DEFAULT_LEAK_SOURCE_FILTER_LITELLM_ONLY,
    DEFAULT_LEAK_SOURCE_MIN_GROWTH_MB,
    DEFAULT_SIGNIFICANT_MEMORY_GROWTH_PERCENT,
    DEFAULT_LEAK_SOURCE_MIN_BATCHES,
    DEFAULT_LEAK_SOURCE_MAX_RESULTS,
)


def parse_file_and_line(file_string: str) -> Optional[Tuple[str, int]]:
    """
    Parse file path and line number from tracemalloc format.
    
    Tracemalloc format: '  File "/path/to/file.py", line 123'
    
    Args:
        file_string: The file string from top_consumers
        
    Returns:
        Tuple of (file_path, line_number) or None if parsing fails
        
    Example:
        >>> parse_file_and_line('  File "/usr/lib/python3.12/pathlib.py", line 404')
        ('/usr/lib/python3.12/pathlib.py', 404)
    """
    # Pattern to match: File "path", line number
    pattern = r'File\s+"([^"]+)",\s+line\s+(\d+)'
    match = re.search(pattern, file_string)
    
    if match:
        file_path = match.group(1)
        line_number = int(match.group(2))
        return (file_path, line_number)
    
    return None


def get_litellm_files_only(file_path: str, litellm_path: str = "/litellm/") -> bool:
    """
    Check if a file path is part of the litellm codebase.
    
    Args:
        file_path: The file path to check
        litellm_path: The path component that identifies litellm files
        
    Returns:
        True if file is in litellm codebase, False otherwise
        
    Example:
        >>> get_litellm_files_only('/Users/alex/Documents/litellm/litellm/types/utils.py')
        True
        >>> get_litellm_files_only('/usr/lib/python3.12/pathlib.py')
        False
    """
    return litellm_path in file_path


def aggregate_memory_by_source(
    snapshots: List[Dict[str, Any]],
    filter_litellm_only: bool = DEFAULT_LEAK_SOURCE_FILTER_LITELLM_ONLY
) -> Dict[str, Dict[int, List[Tuple[int, float]]]]:
    """
    Aggregate memory usage by file and line across all batches.
    
    Args:
        snapshots: List of snapshot dictionaries from JSON file
        filter_litellm_only: If True, only include files from litellm codebase
        
    Returns:
        Dict mapping: file_path -> {line_number -> [(batch, size_mb), ...]}
        
    Example structure:
        {
            '/path/to/file.py': {
                123: [(1, 0.5), (2, 0.6), (3, 0.8)],  # line 123 memory across batches
                456: [(1, 0.2), (2, 0.2), (3, 0.3)]   # line 456 memory across batches
            }
        }
    """
    # Structure: file_path -> line_number -> list of (batch, size_mb)
    memory_by_source: Dict[str, Dict[int, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    
    for snapshot in snapshots:
        batch = snapshot.get('batch')
        top_consumers = snapshot.get('top_consumers', [])
        
        if batch is None or not top_consumers:
            continue
        
        for consumer in top_consumers:
            file_string = consumer.get('file', '')
            size_mb = consumer.get('size_mb', 0.0)
            
            parsed = parse_file_and_line(file_string)
            if not parsed:
                continue
            
            file_path, line_number = parsed
            
            # Filter to litellm files only if requested
            if filter_litellm_only and not get_litellm_files_only(file_path):
                continue
            
            memory_by_source[file_path][line_number].append((batch, size_mb))
    
    return dict(memory_by_source)


def calculate_memory_growth(
    batch_memory_list: List[Tuple[int, float]],
    min_batches: int = 3
) -> Optional[Dict[str, float]]:
    """
    Calculate memory growth for a specific file:line across batches.
    
    Args:
        batch_memory_list: List of (batch, size_mb) tuples sorted by batch
        min_batches: Minimum number of batches required for reliable growth calculation
        
    Returns:
        Dict with growth metrics or None if insufficient data:
        {
            'first_batch': int,
            'last_batch': int,
            'first_size_mb': float,
            'last_size_mb': float,
            'growth_mb': float,
            'growth_percent': float,
            'num_batches': int
        }
    """
    if len(batch_memory_list) < min_batches:
        return None
    
    # Sort by batch number to ensure correct ordering
    sorted_data = sorted(batch_memory_list, key=lambda x: x[0])
    
    first_batch, first_size = sorted_data[0]
    last_batch, last_size = sorted_data[-1]
    
    growth_mb = last_size - first_size
    
    # Calculate growth percentage (handle near-zero cases)
    if first_size > 0.001:  # 1 KB threshold
        growth_percent = (growth_mb / first_size) * 100
    elif growth_mb > 0.001:
        # Started near zero but grew - significant leak
        growth_percent = 999.0  # Cap at 999% for display
    else:
        growth_percent = 0.0
    
    return {
        'first_batch': first_batch,
        'last_batch': last_batch,
        'first_size_mb': first_size,
        'last_size_mb': last_size,
        'growth_mb': growth_mb,
        'growth_percent': growth_percent,
        'num_batches': len(sorted_data)
    }


def identify_leaking_sources(
    memory_by_source: Dict[str, Dict[int, List[Tuple[int, float]]]],
    min_growth_mb: float = DEFAULT_LEAK_SOURCE_MIN_GROWTH_MB,
    min_growth_percent: float = DEFAULT_SIGNIFICANT_MEMORY_GROWTH_PERCENT,
    min_batches: int = DEFAULT_LEAK_SOURCE_MIN_BATCHES
) -> List[Dict[str, Any]]:
    """
    Identify files and lines that are leaking memory.
    
    Args:
        memory_by_source: Output from aggregate_memory_by_source()
        min_growth_mb: Minimum memory growth in MB to consider a leak
        min_growth_percent: Minimum growth percentage to consider a leak
        min_batches: Minimum number of batches required for analysis
        
    Returns:
        List of leak sources, sorted by growth_mb (descending):
        [
            {
                'file_path': str,
                'line_number': int,
                'growth_mb': float,
                'growth_percent': float,
                'first_size_mb': float,
                'last_size_mb': float,
                'first_batch': int,
                'last_batch': int,
                'num_batches': int
            },
            ...
        ]
    """
    leaking_sources = []
    
    for file_path, lines_dict in memory_by_source.items():
        for line_number, batch_memory_list in lines_dict.items():
            growth_metrics = calculate_memory_growth(batch_memory_list, min_batches)
            
            if not growth_metrics:
                continue
            
            # Check if this source meets leak criteria
            if (growth_metrics['growth_mb'] >= min_growth_mb and 
                growth_metrics['growth_percent'] >= min_growth_percent):
                
                leaking_sources.append({
                    'file_path': file_path,
                    'line_number': line_number,
                    'growth_mb': growth_metrics['growth_mb'],
                    'growth_percent': growth_metrics['growth_percent'],
                    'first_size_mb': growth_metrics['first_size_mb'],
                    'last_size_mb': growth_metrics['last_size_mb'],
                    'first_batch': growth_metrics['first_batch'],
                    'last_batch': growth_metrics['last_batch'],
                    'num_batches': growth_metrics['num_batches']
                })
    
    # Sort by growth_mb descending (biggest leaks first)
    leaking_sources.sort(key=lambda x: x['growth_mb'], reverse=True)
    
    return leaking_sources


def load_snapshot_file(output_dir: str, test_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load snapshot data from JSON file for a given test.
    
    Args:
        output_dir: Directory where snapshot files are stored
        test_name: Name of the test
        
    Returns:
        List of snapshot dictionaries or None if file doesn't exist/is invalid
    """
    safe_filename = sanitize_filename(test_name)
    output_file = os.path.join(output_dir, f"{safe_filename}.json")
    
    if not os.path.exists(output_file):
        return None
    
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return None
        
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"[ERROR] Could not load snapshot file {output_file}: {e}")
        return None


def analyze_leaking_sources(
    output_dir: str,
    test_name: str,
    filter_litellm_only: bool = DEFAULT_LEAK_SOURCE_FILTER_LITELLM_ONLY,
    min_growth_mb: float = DEFAULT_LEAK_SOURCE_MIN_GROWTH_MB,
    min_growth_percent: float = DEFAULT_SIGNIFICANT_MEMORY_GROWTH_PERCENT,
    min_batches: int = DEFAULT_LEAK_SOURCE_MIN_BATCHES,
    max_results: int = DEFAULT_LEAK_SOURCE_MAX_RESULTS
) -> Optional[List[Dict[str, Any]]]:
    """
    Complete pipeline to analyze and identify leaking sources from snapshot file.
    
    Args:
        output_dir: Directory where snapshot files are stored
        test_name: Name of the test
        filter_litellm_only: If True, only analyze files from litellm codebase
        min_growth_mb: Minimum memory growth in MB to consider a leak
        min_growth_percent: Minimum growth percentage to consider a leak
        min_batches: Minimum number of batches required for analysis
        max_results: Maximum number of results to return
        
    Returns:
        List of top leaking sources or None if analysis failed
    """
    # Load snapshot data
    snapshots = load_snapshot_file(output_dir, test_name)
    if not snapshots:
        return None
    
    # Aggregate memory by source
    memory_by_source = aggregate_memory_by_source(snapshots, filter_litellm_only)
    
    # Identify leaking sources
    leaking_sources = identify_leaking_sources(
        memory_by_source,
        min_growth_mb,
        min_growth_percent,
        min_batches
    )
    
    # Return top N results
    return leaking_sources[:max_results]


def print_leaking_sources_report(
    leaking_sources: List[Dict[str, Any]],
    test_name: str
) -> None:
    """
    Print a formatted report of leaking sources.
    
    Args:
        leaking_sources: List of leaking source dicts from identify_leaking_sources()
        test_name: Name of the test (for display)
    """
    if not leaking_sources:
        print("\n[LEAK ANALYSIS] No specific leaking sources identified above threshold.")
        return
    
    print("\n" + "=" * 80)
    print("MEMORY LEAK SOURCE ANALYSIS")
    print("=" * 80)
    print(f"Test: {test_name}")
    print(f"Found {len(leaking_sources)} leaking source(s) with significant growth:\n")
    
    for i, source in enumerate(leaking_sources, 1):
        file_path = source['file_path']
        line_number = source['line_number']
        growth_mb = source['growth_mb']
        growth_percent = source['growth_percent']
        first_size = source['first_size_mb']
        last_size = source['last_size_mb']
        first_batch = source['first_batch']
        last_batch = source['last_batch']
        num_batches = source['num_batches']
        
        # Format growth percent (cap display at 999%)
        growth_pct_str = f"{min(growth_percent, 999.0):.1f}%"
        if growth_percent > 999.0:
            growth_pct_str += "+"
        
        print(f"{i}. {file_path}:{line_number}")
        print(f"   Growth: {growth_mb:.3f} MB ({growth_pct_str})")
        print(f"   Batch {first_batch}: {first_size:.3f} MB → Batch {last_batch}: {last_size:.3f} MB")
        print(f"   Tracked across {num_batches} batch(es)")
        print()
    
    print("=" * 80)
    print("RECOMMENDATION: Investigate the above files/lines for memory leaks.")
    print("Look for:")
    print("  • Objects not being properly released")
    print("  • Caches growing unbounded")
    print("  • Event listeners/callbacks not being cleaned up")
    print("  • Circular references preventing garbage collection")
    print("=" * 80 + "\n")


def analyze_and_report_leaking_sources(
    output_dir: str,
    test_name: str,
    filter_litellm_only: bool = DEFAULT_LEAK_SOURCE_FILTER_LITELLM_ONLY,
    min_growth_mb: float = DEFAULT_LEAK_SOURCE_MIN_GROWTH_MB,
    min_growth_percent: float = DEFAULT_SIGNIFICANT_MEMORY_GROWTH_PERCENT,
    min_batches: int = DEFAULT_LEAK_SOURCE_MIN_BATCHES,
    max_results: int = DEFAULT_LEAK_SOURCE_MAX_RESULTS
) -> None:
    """
    Convenience function to analyze and print leaking sources report.
    
    Args:
        output_dir: Directory where snapshot files are stored
        test_name: Name of the test
        filter_litellm_only: If True, only analyze files from litellm codebase
        min_growth_mb: Minimum memory growth in MB to consider a leak
        min_growth_percent: Minimum growth percentage to consider a leak
        min_batches: Minimum number of batches required for analysis
        max_results: Maximum number of results to return
    """
    leaking_sources = analyze_leaking_sources(
        output_dir,
        test_name,
        filter_litellm_only,
        min_growth_mb,
        min_growth_percent,
        min_batches,
        max_results
    )
    
    if leaking_sources is None:
        print(f"\n[LEAK ANALYSIS] Could not load snapshot data for test '{test_name}'")
        print("[LEAK ANALYSIS] Make sure capture_top_consumers was enabled during the test.")
        return
    
    print_leaking_sources_report(leaking_sources, test_name)

