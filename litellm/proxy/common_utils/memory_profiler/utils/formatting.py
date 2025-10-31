"""
Output formatting utilities for memory leak testing.

Provides functions for printing formatted output including:
- Headers and separators
- Memory metrics and statistics
- Memory sample summaries
"""

from typing import Dict, List


def print_analysis_header(title: str = "Memory Growth Analysis") -> None:
    """
    Print a formatted analysis header.
    
    Args:
        title: Title to display in the header
        
    Example:
        >>> print_analysis_header("Custom Analysis")
        
        ======================================================================
        Custom Analysis
        ======================================================================
    """
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)


def print_test_header(title: str = "Memory Leak Detection Test") -> None:
    """
    Print a formatted test header.
    
    Args:
        title: Title to display in the header
        
    Example:
        >>> print_test_header("My Test")
        
        ======================================================================
        My Test
        ======================================================================
    """
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)


def print_growth_metrics(growth_metrics: Dict[str, float]) -> None:
    """
    Print formatted growth metrics.
    
    Args:
        growth_metrics: Dictionary containing 'initial_avg', 'final_avg', 
                       'growth', and 'growth_percent' keys
                       
    Example:
        >>> metrics = {
        ...     'initial_avg': 10.5,
        ...     'final_avg': 12.3,
        ...     'growth': 1.8,
        ...     'growth_percent': 17.1
        ... }
        >>> print_growth_metrics(metrics)
        Initial avg: 10.50 MB
        Final avg:   12.30 MB
        Growth:      1.80 MB (17.1%)
    """
    print(f"Initial avg: {growth_metrics['initial_avg']:.2f} MB")
    print(f"Final avg:   {growth_metrics['final_avg']:.2f} MB")
    print(f"Growth:      {growth_metrics['growth']:.2f} MB "
          f"({growth_metrics['growth_percent']:.1f}%)")


def print_memory_samples(memory_samples: List[float], num_samples: int = 10) -> None:
    """
    Print the last N memory samples.
    
    Args:
        memory_samples: List of memory measurements in MB
        num_samples: Number of samples to display (from the end)
        
    Example:
        >>> samples = [10.1, 10.2, 10.3, 10.5, 10.4]
        >>> print_memory_samples(samples, num_samples=3)
        Samples (last 3): ['10.30MB', '10.50MB', '10.40MB']
        ======================================================================
    """
    samples_str = [f'{m:.2f}MB' for m in memory_samples[-num_samples:]]
    print(f"Samples (last {num_samples}): {samples_str}")
    print("="*70)

