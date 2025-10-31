"""
Analyze endpoint memory profile data for memory leaks and growth patterns.

This module provides functions to analyze collected profile data using
algorithms from tests.memory_leak_tests for leak detection.

Features:
- Memory growth analysis
- Memory leak detection
- Top memory consumers aggregation
- Request-by-request memory tracking

Usage:
    from tests.memory_leak_tests.profiler.analyze_profiles import (
        load_profile_data,
        analyze_endpoint_memory,
        print_analysis_report
    )
    
    profiles = load_profile_data("endpoint_profiles/chat_completions.json")
    analysis = analyze_endpoint_memory(profiles)
    print_analysis_report(analysis, "chat_completions")
"""

import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Add project root to path for imports (needed when running as script)
# Get the litellm root directory (5 levels up from this file)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import memory leak detection from tests
try:
    from tests.memory_leak_tests.analysis.detection import detect_memory_leak
    from tests.memory_leak_tests.analysis.growth import (
        analyze_memory_growth,
        prepare_memory_analysis,
    )
    _HAS_MEMORY_LEAK_TESTS = True
except ImportError:
    _HAS_MEMORY_LEAK_TESTS = False
    # Define None to satisfy linter - these won't be used if _HAS_MEMORY_LEAK_TESTS is False
    detect_memory_leak = None  # type: ignore
    analyze_memory_growth = None  # type: ignore
    prepare_memory_analysis = None  # type: ignore


# ============================================================================
# Constants
# ============================================================================

# Memory analysis parameters
DEFAULT_SAMPLE_WINDOW = 3  # Number of samples for rolling average
DEFAULT_SAMPLE_SIZE = 3  # Number of samples for initial/final averages
LEAK_THRESHOLD_PERCENT = 25.0  # Growth percentage threshold for leak detection

# Display/reporting parameters
DEFAULT_TOP_N = 20  # Default number of top items to display
DEFAULT_GROWTH_DISPLAY = 20  # Default number of growing locations to show

# Output formatting
SEPARATOR_WIDTH = 80  # Width of separator lines in reports


# ============================================================================
# Data Loading
# ============================================================================


def load_profile_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load profile data from JSON file.
    
    Args:
        file_path: Path to profile JSON file
        
    Returns:
        List of profile dictionaries
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {file_path}, got {type(data)}")
    
    return data


def extract_request_number(request_id: str) -> int:
    """
    Extract numeric request number from request_id.
    
    Args:
        request_id: Request ID like "req-123"
        
    Returns:
        Numeric part (123)
    """
    try:
        return int(request_id.split('-')[1])
    except (IndexError, ValueError):
        return 0


# ============================================================================
# Memory Analysis
# ============================================================================


def extract_memory_samples(profiles: List[Dict[str, Any]]) -> tuple[List[float], List[int]]:
    """
    Extract memory samples and error counts from profiles.
    
    Args:
        profiles: List of profile dictionaries
        
    Returns:
        Tuple of (memory_samples, error_counts) - one per profile
    """
    memory_samples = []
    error_counts = []
    
    sorted_profiles = sorted(
        profiles,
        key=lambda p: extract_request_number(p.get('request_id', 'req-0'))
    )
    
    for profile in sorted_profiles:
        if 'memory' in profile and 'current_mb' in profile['memory']:
            memory_samples.append(profile['memory']['current_mb'])
            error_counts.append(1 if profile.get('had_error', False) else 0)
    
    return memory_samples, error_counts


def analyze_endpoint_memory(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze memory growth and detect leaks for an endpoint.
    
    This function uses algorithms from tests.memory_leak_tests for
    memory leak detection.
    
    Args:
        profiles: List of profile dictionaries
        
    Returns:
        Dictionary with analysis results including leak detection
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
    
    if not _HAS_MEMORY_LEAK_TESTS:
        # Fallback to basic analysis if memory_leak_tests not available
        return _basic_memory_analysis(memory_samples, error_counts)
    
    # Use memory leak detection from tests
    try:
        # Type assertions for linter - we know these are available when _HAS_MEMORY_LEAK_TESTS is True
        assert prepare_memory_analysis is not None
        assert analyze_memory_growth is not None
        assert detect_memory_leak is not None
        
        # Prepare memory data
        rolling_avg, num_samples, tail_samples = prepare_memory_analysis(
            memory_samples,
            sample_window=DEFAULT_SAMPLE_WINDOW
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
            'has_advanced_detection': True,
        }
        
    except Exception as e:
        return {
            'error': f'Error during analysis: {str(e)}'
        }


def _basic_memory_analysis(
    memory_samples: List[float],
    error_counts: List[int]
) -> Dict[str, Any]:
    """
    Fallback basic memory analysis when memory_leak_tests not available.
    
    Args:
        memory_samples: List of memory measurements
        error_counts: List of error counts per request
        
    Returns:
        Basic analysis results
    """
    initial = statistics.mean(memory_samples[:DEFAULT_SAMPLE_SIZE]) if len(memory_samples) >= DEFAULT_SAMPLE_SIZE else memory_samples[0]
    final = statistics.mean(memory_samples[-DEFAULT_SAMPLE_SIZE:]) if len(memory_samples) >= DEFAULT_SAMPLE_SIZE else memory_samples[-1]
    growth = final - initial
    growth_percent = (growth / initial * 100) if initial > 0 else 0
    
    leak_detected = growth_percent > LEAK_THRESHOLD_PERCENT
    leak_message = f"Memory grew by {growth_percent:.1f}%" if leak_detected else "No significant growth"
    
    return {
        'total_requests': len(memory_samples),
        'memory_samples': memory_samples,
        'error_counts': error_counts,
        'growth_metrics': {
            'initial_avg': initial,
            'final_avg': final,
            'growth': growth,
            'growth_percent': growth_percent,
        },
        'leak_detected': leak_detected,
        'leak_message': leak_message,
        'has_advanced_detection': False,
    }


# ============================================================================
# Top Memory Consumers Analysis
# ============================================================================


def analyze_top_memory_consumers(profiles: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N) -> None:
    """
    Aggregate and analyze top memory consumers across all requests.
    
    Args:
        profiles: List of profile dictionaries
        top_n: Number of top consumers to show
    """
    # Aggregate memory by file:line
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
            
            # Clean up file info
            location = file_info
            if 'File' in file_info and 'line' in file_info:
                parts = file_info.split('"')
                if len(parts) >= 2:
                    file_path = parts[1]
                    line_parts = file_info.split('line ')
                    line_num = line_parts[1].split(',')[0] if len(line_parts) > 1 else '?'
                    location = f"{file_path}:{line_num}"
            
            memory_by_location[location]['total_mb'] += size_mb
            memory_by_location[location]['max_mb'] = max(
                memory_by_location[location]['max_mb'], size_mb
            )
            memory_by_location[location]['count'] += 1
            memory_by_location[location]['samples'].append(size_mb)
    
    # Sort by total memory
    sorted_locations = sorted(
        memory_by_location.items(),
        key=lambda x: x[1]['total_mb'],
        reverse=True
    )[:top_n]
    
    print("\n" + "="*SEPARATOR_WIDTH)
    print(f"TOP {top_n} MEMORY-CONSUMING LOCATIONS (Aggregated)")
    print("="*SEPARATOR_WIDTH)
    print(f"{'Rank':<6} {'Total (MB)':<12} {'Max (MB)':<12} {'Avg (MB)':<12} {'Samples':<10} {'Location'}")
    print("-"*SEPARATOR_WIDTH)
    
    for rank, (location, data) in enumerate(sorted_locations, 1):
        avg_mb = data['total_mb'] / data['count']
        print(f"{rank:<6} {data['total_mb']:<12.3f} {data['max_mb']:<12.3f} {avg_mb:<12.3f} {data['count']:<10} {location}")
    
    print("="*SEPARATOR_WIDTH + "\n")


def analyze_memory_growth_by_location(profiles: List[Dict[str, Any]], top_n: int = DEFAULT_TOP_N) -> None:
    """
    Analyze memory growth by location (first vs last appearance).
    
    Shows which files/lines are growing in memory over time.
    
    Args:
        profiles: List of profile dictionaries
        top_n: Number of top growing locations to show
    """
    # Sort profiles by request ID
    sorted_profiles = sorted(
        profiles,
        key=lambda p: extract_request_number(p.get('request_id', 'req-0'))
    )
    
    # Track first and last appearance of each location
    location_data: Dict[str, Dict[str, Any]] = {}
    
    for idx, profile in enumerate(sorted_profiles):
        if 'memory' not in profile or 'top_consumers' not in profile['memory']:
            continue
        
        for consumer in profile['memory']['top_consumers']:
            file_info = consumer['file'].strip()
            size_mb = consumer['size_mb']
            
            # Clean up file info
            location = file_info
            if 'File' in file_info and 'line' in file_info:
                parts = file_info.split('"')
                if len(parts) >= 2:
                    file_path = parts[1]
                    line_parts = file_info.split('line ')
                    line_num = line_parts[1].split(',')[0] if len(line_parts) > 1 else '?'
                    location = f"{file_path}:{line_num}"
            
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
    
    # Calculate growth for each location
    growth_data = []
    for location, data in location_data.items():
        if data['first_idx'] == data['last_idx']:
            continue  # Skip locations that only appeared once
        
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
    
    # Sort by absolute growth
    sorted_growth = sorted(growth_data, key=lambda x: abs(x['growth_mb']), reverse=True)[:top_n]
    
    if not sorted_growth:
        print("\n" + "="*SEPARATOR_WIDTH)
        print("MEMORY GROWTH BY LOCATION")
        print("="*SEPARATOR_WIDTH)
        print("No locations with multiple samples found for analysis.")
        print("="*SEPARATOR_WIDTH + "\n")
        return
    
    print("\n" + "="*SEPARATOR_WIDTH)
    print(f"TOP {top_n} LOCATIONS BY MEMORY GROWTH (First → Last)")
    print("="*SEPARATOR_WIDTH)
    print(f"{'Rank':<6} {'Growth (MB)':<14} {'Growth %':<12} {'Range (MB)':<20} {'Samples':<10} {'Location'}")
    print("-"*SEPARATOR_WIDTH)
    
    for rank, data in enumerate(sorted_growth, 1):
        range_str = f"{data['min_mb']:.2f} → {data['max_mb']:.2f}"
        location = data['location']
        
        print(f"{rank:<6} {data['growth_mb']:<14.3f} {data['growth_percent']:<12.1f} {range_str:<20} {data['sample_count']:<10} {location}")
    
    print("="*SEPARATOR_WIDTH + "\n")


# ============================================================================
# Reporting
# ============================================================================


def print_analysis_report(analysis: Dict[str, Any], endpoint_name: str) -> None:
    """
    Print formatted analysis report for an endpoint.
    
    Args:
        analysis: Analysis results from analyze_endpoint_memory()
        endpoint_name: Name of the endpoint
    """
    print("\n" + "="*SEPARATOR_WIDTH)
    print(f"ENDPOINT MEMORY ANALYSIS: {endpoint_name}")
    print("="*SEPARATOR_WIDTH)
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}\n")
        return
    
    print(f"Total Requests Analyzed: {analysis['total_requests']}")
    
    # Print growth metrics
    if 'growth_metrics' in analysis:
        metrics = analysis['growth_metrics']
        print(f"\nMemory Growth:")
        print(f"  Initial Average: {metrics['initial_avg']:.3f} MB")
        print(f"  Final Average:   {metrics['final_avg']:.3f} MB")
        print(f"  Growth:          {metrics['growth']:.3f} MB ({metrics['growth_percent']:.1f}%)")
    
    # Print leak detection results
    print(f"\n{'='*SEPARATOR_WIDTH}")
    if analysis.get('leak_detected', False):
        print("MEMORY LEAK DETECTED")
        print("="*SEPARATOR_WIDTH)
        print(f"Message: {analysis.get('leak_message', 'Unknown')}")
        print("\nRecommendations:")
        print("  - Review memory-consuming code paths")
        print("  - Check for unbounded caches or collections")
        print("  - Verify proper cleanup of resources")
        print("  - Run detailed analysis on this endpoint")
    else:
        print("NO MEMORY LEAK DETECTED")
        print("="*SEPARATOR_WIDTH)
        print(f"Message: {analysis.get('leak_message', 'Memory appears stable')}")
    
    if not analysis.get('has_advanced_detection', False):
        print("\nNote: Using fallback analysis (advanced leak detection not available)")
        print("   To use advanced leak detection, run this script as a module:")
        print("   python -m tests.memory_leak_tests.profiler.analyze_profiles <file>")
    
    print("="*SEPARATOR_WIDTH + "\n")


# ============================================================================
# Main Analysis Function
# ============================================================================


def analyze_profile_file(
    profile_file: str,
    show_growth: Optional[int] = DEFAULT_GROWTH_DISPLAY,
) -> Dict[str, Any]:
    """
    Analyze a profile file and print results.
    
    Args:
        profile_file: Path to profile JSON file
        show_growth: Number of growing locations to show (default: DEFAULT_GROWTH_DISPLAY, None = don't show)
        
    Returns:
        Dictionary with analysis results
        
    Example:
        >>> results = analyze_profile_file("endpoint_profiles/chat_completions.json")
        >>> if results.get('leak_detected'):
        ...     print("Leak found!")
    """
    # Load data
    print(f"\nLoading profile data from: {profile_file}")
    try:
        profiles = load_profile_data(profile_file)
        print(f"Loaded {len(profiles)} profile entries\n")
    except Exception as e:
        print(f"Error loading profile data: {e}\n")
        return {'error': str(e)}
    
    if not profiles:
        print("No profiles found in file\n")
        return {'error': 'No profiles found'}
    
    # Extract endpoint name from filename
    import os
    endpoint_name = os.path.basename(profile_file).replace('.json', '')
    
    # Analyze memory
    analysis = analyze_endpoint_memory(profiles)
    print_analysis_report(analysis, endpoint_name)
    
    # Show memory growth by location (default)
    if show_growth:
        analyze_memory_growth_by_location(profiles, top_n=show_growth)
    
    return analysis


# ============================================================================
# Command-line entry point
# ============================================================================


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m tests.memory_leak_tests.profiler.analyze_profiles <profile_file.json> [--growth N]")
        print("\nOptions:")
        print("  --growth N    Show top N locations by memory growth (default: 20)")
        print("\nExamples:")
        print("  # Show default growth analysis (top 20)")
        print("  python -m tests.memory_leak_tests.profiler.analyze_profiles endpoint_profiles/chat_completions.json")
        print("\n  # Show top 30 growing locations")
        print("  python -m tests.memory_leak_tests.profiler.analyze_profiles endpoint_profiles/chat_completions.json --growth 30")
        print("\nNote: Run from the project root directory")
        sys.exit(1)
    
    profile_file = sys.argv[1]
    growth_n = DEFAULT_GROWTH_DISPLAY  # Default
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--growth' and i + 1 < len(sys.argv):
            growth_n = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    analyze_profile_file(profile_file, show_growth=growth_n)
