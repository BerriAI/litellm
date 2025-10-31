"""
Command-line entry point for endpoint memory profile analysis.

This allows running the analyzer as:
    python -m litellm.proxy.common_utils.memory_profiler.endpoint_profiler <profile_file.json>

Instead of:
    python -m litellm.proxy.common_utils.memory_profiler.endpoint_profiler.analyze_profiles <profile_file.json>

The modular analysis system provides:
- Data loading (data_loading.py)
- Memory analysis (memory_analysis.py)
- Consumer analysis (consumer_analysis.py)
- Location analysis (location_analysis.py)
- Reporting (reporting.py)
- Orchestration (analyze_profiles.py)
"""

from .analyze_profiles import main

if __name__ == '__main__':
    main()

