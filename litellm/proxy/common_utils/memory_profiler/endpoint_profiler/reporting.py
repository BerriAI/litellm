"""
Reporting utilities for endpoint memory profiling.

This module provides functions for formatting and printing analysis reports.

Features:
- Format analysis results as readable reports
- Print leak detection results
- Display growth metrics
"""

from typing import Any, Dict


# Output formatting
SEPARATOR_WIDTH = 80  # Width of separator lines in reports


def print_analysis_report(analysis: Dict[str, Any], endpoint_name: str) -> None:
    """
    Print formatted analysis report for an endpoint.
    
    Displays:
    - Total requests analyzed
    - Memory growth metrics
    - Leak detection results
    - Recommendations (if leak detected)
    
    Args:
        analysis: Analysis results from analyze_endpoint_memory()
        endpoint_name: Name of the endpoint
        
    Example:
        >>> analysis = analyze_endpoint_memory(profiles)
        >>> print_analysis_report(analysis, "chat_completions")
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
    
    print("="*SEPARATOR_WIDTH + "\n")


def print_loading_info(profile_file: str, profile_count: int) -> None:
    """
    Print information about loaded profile data.
    
    Args:
        profile_file: Path to profile file
        profile_count: Number of profiles loaded
        
    Example:
        >>> print_loading_info("endpoint_profiles/chat.json", 100)
    """
    print(f"\nLoading profile data from: {profile_file}")
    print(f"Loaded {profile_count} profile entries\n")


def print_error(error_message: str) -> None:
    """
    Print error message.
    
    Args:
        error_message: Error message to display
        
    Example:
        >>> print_error("File not found")
    """
    print(f"Error: {error_message}\n")

