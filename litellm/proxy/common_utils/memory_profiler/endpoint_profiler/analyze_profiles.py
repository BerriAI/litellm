"""
Analyze endpoint memory profile data for memory leaks and growth patterns.

This module orchestrates profile analysis by combining specialized modules:
- data_loading: Load profile data from files
- memory_analysis: Core memory growth and leak detection
- consumer_analysis: Aggregate top memory consumers
- location_analysis: Track memory growth by code location
- reporting: Format and print analysis reports

Usage:
    from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import (
        load_profile_data,
        analyze_endpoint_memory,
        analyze_profile_file,
    )
    
    profiles = load_profile_data("endpoint_profiles/chat_completions.json")
    analysis = analyze_endpoint_memory(profiles)
"""

import os
from typing import Any, Dict, Optional

# Import from specialized modules
from .data_loading import load_profile_data
from .memory_analysis import analyze_endpoint_memory
from .location_analysis import analyze_memory_growth_by_location
from .reporting import print_analysis_report, print_loading_info, print_error


# ============================================================================
# Constants
# ============================================================================

# Display/reporting parameters
DEFAULT_GROWTH_DISPLAY = 20  # Default number of growing locations to show


# ============================================================================
# Main Analysis Function
# ============================================================================


def analyze_profile_file(
    profile_file: str,
    show_growth: Optional[int] = DEFAULT_GROWTH_DISPLAY,
) -> Dict[str, Any]:
    """
    Analyze a profile file and print results.
    
    Orchestrates the complete analysis workflow:
    1. Load profile data from file
    2. Analyze memory growth and detect leaks
    3. Print formatted report
    4. Optionally analyze memory growth by location
    
    Args:
        profile_file: Path to profile JSON file
        show_growth: Number of growing locations to show (default: DEFAULT_GROWTH_DISPLAY, None = don't show)
        
    Returns:
        Dictionary with analysis results including:
        - total_requests: Number of requests analyzed
        - memory_samples: Raw memory data
        - growth_metrics: Memory growth metrics
        - leak_detected: Boolean indicating if leak found
        - leak_message: Description of findings
        - error: Error message (if analysis failed)
        
    Example:
        >>> results = analyze_profile_file("endpoint_profiles/chat_completions.json")
        >>> if results.get('leak_detected'):
        ...     print("Leak found!")
    """
    # Load data
    try:
        profiles = load_profile_data(profile_file)
        print_loading_info(profile_file, len(profiles))
    except Exception as e:
        print_error(f"Error loading profile data: {e}")
        return {'error': str(e)}
    
    if not profiles:
        print_error("No profiles found in file")
        return {'error': 'No profiles found'}
    
    # Extract endpoint name from filename
    endpoint_name = os.path.basename(profile_file).replace('.json', '')
    
    # Analyze memory
    analysis = analyze_endpoint_memory(profiles)
    print_analysis_report(analysis, endpoint_name)
    
    # Show memory growth by location (if requested)
    if show_growth:
        analyze_memory_growth_by_location(profiles, top_n=show_growth)
    
    return analysis


# ============================================================================
# Command-line entry point
# ============================================================================


def main():
    """
    Command-line entry point for profile analysis.
    
    Parses arguments and runs analysis with user-specified options.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m litellm.proxy.common_utils.memory_profiler.endpoint_profiler.analyze_profiles <profile_file.json> [--growth N]")  # noqa: T201
        print("\nOptions:")  # noqa: T201
        print("  --growth N    Show top N locations by memory growth (default: 20)")  # noqa: T201
        print("\nExamples:")  # noqa: T201
        print("  # Show default growth analysis (top 20)")  # noqa: T201
        print("  python -m litellm.proxy.common_utils.memory_profiler.endpoint_profiler.analyze_profiles endpoint_profiles/chat_completions.json")  # noqa: T201
        print("\n  # Show top 30 growing locations")  # noqa: T201
        print("  python -m litellm.proxy.common_utils.memory_profiler.endpoint_profiler.analyze_profiles endpoint_profiles/chat_completions.json --growth 30")  # noqa: T201
        print("\nNote: Run from the project root directory")  # noqa: T201
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


if __name__ == '__main__':
    main()
