"""
Memory analysis and leak detection system.

This package provides functionality for:
- Checking measurement noise and stability
- Analyzing memory growth patterns
- Detecting memory leaks (including error-induced leaks)
- Identifying specific leaking sources (files and lines)
"""

from .detection import (
    detect_error_induced_memory_leak,
    detect_memory_leak,
)
from .growth import (
    prepare_memory_analysis,
    calculate_rolling_average,
    analyze_memory_growth,
)
from .noise import check_measurement_noise
from .leaking_sources import (
    analyze_leaking_sources,
    analyze_and_report_leaking_sources,
    identify_leaking_sources,
    aggregate_memory_by_source,
    parse_file_and_line,
    print_leaking_sources_report,
)

__all__ = [
    # Noise detection
    "check_measurement_noise",
    # Growth analysis
    "prepare_memory_analysis",
    "calculate_rolling_average",
    "analyze_memory_growth",
    # Leak detection
    "detect_error_induced_memory_leak",
    "detect_memory_leak",
    # Leak source identification
    "analyze_leaking_sources",
    "analyze_and_report_leaking_sources",
    "identify_leaking_sources",
    "aggregate_memory_by_source",
    "parse_file_and_line",
    "print_leaking_sources_report",
]

