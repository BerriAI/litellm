# Memory Leak Analysis

This package provides comprehensive memory leak detection and source identification.

## Features

### 1. Memory Leak Detection (`detection.py`)

- Detects error-induced memory leaks
- Detects general memory leaks based on growth patterns
- Checks for continuous memory growth

### 2. Memory Growth Analysis (`growth.py`)

- Calculates rolling averages
- Analyzes memory growth patterns
- Prepares memory data for leak detection

### 3. Measurement Noise Detection (`noise.py`)

- Checks if measurements are too noisy for reliable analysis
- Calculates coefficient of variation

### 4. **Leaking Source Identification (`leaking_sources.py`)**

Automatically identifies specific files and line numbers that are leaking memory when a test detects a memory leak.

#### How It Works

When a memory leak is detected:

1. Loads the memory snapshot JSON file for the test
2. Aggregates memory usage by file path and line number across all batches
3. Calculates memory growth for each file:line combination
4. Identifies sources with significant growth
5. Reports the top leaking sources with exact file paths and line numbers

#### Output Example

```
================================================================================
MEMORY LEAK SOURCE ANALYSIS
================================================================================
Test: Router acompletion (streaming) - Memory Leak Detection Test
Found 1 leaking source(s) with significant growth:

1. /path/to/litellm/litellm/types/utils.py:1143
   Growth: 1.234 MB (325.6%)
   Batch 1: 0.379 MB → Batch 10: 1.613 MB
   Tracked across 63 batch(es)

================================================================================
RECOMMENDATION: Investigate the above files/lines for memory leaks.
Look for:
  • Objects not being properly released
  • Caches growing unbounded
  • Event listeners/callbacks not being cleaned up
  • Circular references preventing garbage collection
================================================================================
```

#### Configuration

The analysis is automatically run when a leak is detected with these default settings:

- **filter_litellm_only**: `True` - Only analyzes files from the litellm codebase
- **min_growth_mb**: `0.1` - Minimum 0.1 MB growth to report
- **min_growth_percent**: `50.0%` - Minimum 50% growth to report
- **min_batches**: `3` - Must be present in at least 3 batches
- **max_results**: `10` - Report top 10 leaking sources

#### Manual Usage

You can also run the analysis manually:

```python
from tests.memory_leak_tests.analysis import analyze_and_report_leaking_sources

# Analyze a specific test's snapshots
analyze_and_report_leaking_sources(
    output_dir="memory_snapshots",
    test_name="My Test Name",
    filter_litellm_only=True,  # False to include stdlib files
    min_growth_mb=0.1,
    min_growth_percent=50.0,
    min_batches=3,
    max_results=10
)
```

#### Functions

- **parse_file_and_line()** - Parses tracemalloc file strings
- **aggregate_memory_by_source()** - Aggregates memory by file:line across batches
- **calculate_memory_growth()** - Calculates growth metrics for a source
- **identify_leaking_sources()** - Identifies sources meeting leak criteria
- **analyze_leaking_sources()** - Complete pipeline from file to results
- **print_leaking_sources_report()** - Formats and prints results
- **analyze_and_report_leaking_sources()** - Convenience function (analyze + print)

## Integration

The leaking source analysis is automatically integrated into `analyze_and_detect_leaks()` in `core/execution.py`. When a memory leak is detected:

1. The regular leak detection runs
2. **If a leak is found**, the source analysis runs automatically
3. A detailed report is printed showing exactly which files/lines are leaking
4. The test fails with the leak message

This provides immediate, actionable information about where to fix the memory leak.

## Requirements

- Memory snapshots must be captured with `capture_top_consumers=True` (default)
- Tests must run with the default `output_dir="memory_snapshots"` (default)
- Snapshot JSON files must exist in the output directory

## Benefits

- **Pinpoint accuracy** - Know exactly which file and line is leaking
- **Actionable insights** - Get specific memory growth amounts and percentages
- **Automatic integration** - No manual steps required
- **Cross-batch tracking** - See how memory grows across the entire test
- **Filtered results** - Focus on litellm code or include all sources
