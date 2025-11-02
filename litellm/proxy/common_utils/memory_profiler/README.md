# Memory Profiler Library

A comprehensive library for memory leak detection and profiling during development and testing.

## What is This?

This package provides tools for **detecting memory leaks in development/testing environments**:

- **Core utilities**: Memory analysis, leak detection, and snapshot management
- **Endpoint profiler**: Memory monitoring for development/testing
- **Reusable components**: Used by tests in `tests/memory_leak_tests/`

**Note**: This is a **development and testing tool**. Not intended for production monitoring.

## Quick Start

### For Testing

This package is used by memory leak tests in `tests/memory_leak_tests/`.

```bash
# Run memory leak tests
pytest tests/memory_leak_tests/

# Run specific test
pytest tests/memory_leak_tests/test_sdk_completion.py
```

### For Development Profiling

```python
from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import (
    EndpointProfiler,
    profile_endpoint
)

# Initialize for development/testing
profiler = EndpointProfiler.get_instance(enabled=True, sampling_rate=1.0)
await profiler.start_auto_flush()

# Decorate endpoints to profile during development
@profile_endpoint()
async def chat_completions(request: Request):
    return {"response": "data"}

# Analyze collected profiles
python -m litellm.proxy.common_utils.memory_profiler.endpoint_profiler endpoint_profiles/chat_completions.json
```

See `endpoint_profiler/README.md` for complete profiler documentation.

## Package Structure

The package is organized into modular components:

### Core Modules

- `constants.py` - Configuration (batch size, thresholds, smart capture settings)
- `core/` - Core execution logic
  - `execution.py` - Test execution engine
  - `config.py` - Configuration management
  - `cleanup.py` - Resource cleanup utilities

### Analysis Modules

- `analysis/` - Memory analysis and leak detection
  - `detection.py` - Leak detection algorithms
  - `growth.py` - Memory growth analysis
  - `noise.py` - Noise detection and filtering
  - `leaking_sources.py` - Identifies sources of memory leaks
  - `README.md` - Detailed analysis documentation

### Snapshot Management

- `snapshot/` - Memory snapshot capture and storage
  - `capture.py` - Smart memory snapshot capturing
  - `storage.py` - Snapshot persistence

### Development Profiler

- `endpoint_profiler/` - Endpoint memory monitoring for development/testing
  - **Profiler modules (modular)**:
    - `profiler_core.py` - EndpointProfiler singleton class
    - `decorator.py` - profile_endpoint decorator
    - `profiler.py` - Re-exports for backward compatibility
  - `capture.py` - Request-level memory capture
  - `storage.py` - Profile buffering and persistence
  - **Analysis modules (modular)**:
    - `analyze_profiles.py` - Main orchestration layer
    - `data_loading.py` - Profile data loading utilities
    - `memory_analysis.py` - Core leak detection and growth analysis
    - `consumer_analysis.py` - Top memory consumers analysis
    - `location_analysis.py` - Memory growth by code location
    - `reporting.py` - Formatted output and reports
  - `utils.py` - Helper functions
  - `constants.py` - Configuration constants
  - `README.md` - Complete profiler documentation

### Utilities

- `utils/` - Helper functions
  - `conversions.py` - Data type conversions
  - `formatting.py` - Output formatting

## Key Features

### Smart Capture

Captures detailed memory snapshots selectively to balance detail with performance:

- **First request** - Always (baseline)
- **On errors** - Always (to debug error-related leaks)
- **Memory increases** - When memory grows by >2% (configurable)
- **Periodic sampling** - Every N requests (configurable)

### Professional Leak Detection

Uses battle-tested algorithms:

- Linear regression analysis
- Statistical significance testing
- Error-induced leak detection
- Variance filtering

### Modular Design

Each component has a single responsibility:

- Easy to test individually
- Can be imported independently
- Clear separation of concerns

## Configuration

Edit `constants.py` to adjust test parameters:

```python
# Execution speed controls
DEFAULT_NUM_BATCHES = 6                              # Number of measurement batches
DEFAULT_BATCH_SIZE = 50                              # Requests per batch
DEFAULT_WARMUP_BATCHES = 2                           # Warmup batches before measurement
DEFAULT_LEAK_DETECTION_MAX_GROWTH_PERCENT = 25.0    # Leak detection threshold

# Smart capture configuration
DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT = 2.0          # Capture when memory grows by >2%
DEFAULT_PERIODIC_SAMPLE_INTERVAL = 25                # Capture every 25 requests
DEFAULT_TOP_CONSUMERS_COUNT = 10                     # Top memory consumers to track
```

## Usage Examples

### Running Tests

```python
from litellm.proxy.common_utils.memory_profiler import (
    run_memory_measurement_with_tracemalloc,
    analyze_and_detect_leaks,
    get_memory_test_config,
)

# Configure and run test
config = get_memory_test_config(batch_size=50, num_batches=6)
memory_samples, error_counts = await run_memory_measurement_with_tracemalloc(...)
analyze_and_detect_leaks(memory_samples, error_counts, config)
```

### Memory Analysis

```python
from litellm.proxy.common_utils.memory_profiler import (
    analyze_memory_growth,
    detect_memory_leak,
)

# Analyze memory samples
growth_metrics = analyze_memory_growth(rolling_avg, num_samples)
leak_detected, message = detect_memory_leak(growth_metrics, memory_samples, error_counts)
```

### Endpoint Profiling (Development)

```python
from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import (
    analyze_profile_file,
    load_profile_data,
)

# Load and analyze profile data
profiles = load_profile_data("endpoint_profiles/chat_completions.json")
analysis = analyze_profile_file("endpoint_profiles/chat_completions.json")

if analysis.get('leak_detected'):
    print(f"Leak detected: {analysis['leak_message']}")
```

### Custom Analysis Workflow

```python
from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import (
    load_profile_data,
    analyze_endpoint_memory,
    analyze_top_memory_consumers,
    analyze_memory_growth_by_location,
)

# Load data
profiles = load_profile_data("endpoint_profiles/chat.json")

# Run specific analyses
memory_analysis = analyze_endpoint_memory(profiles)
analyze_top_memory_consumers(profiles, top_n=10)
analyze_memory_growth_by_location(profiles, top_n=20)
```

## How Memory Tests Work

1. **Warmup** (2 batches): Let caches stabilize
2. **Measurement** (6 batches): Run requests and measure memory
3. **Analysis**: Check if memory grows or stays stable

Each batch = 50 API requests. Total: 400 requests per test (optimized for speed).

## Documentation

- `endpoint_profiler/README.md` - Complete endpoint profiler documentation
- `analysis/README.md` - Detailed analysis algorithms documentation
- Individual module docstrings - Inline documentation

## Performance

- **400 requests per test** (optimized from 1,300)
- **Smart snapshot capture** - detailed profiling only when needed
- **~69% faster** than previous configuration
- Test suite runs in ~5-8 minutes

## Best Practices for Testing

1. **Run tests regularly** - Catch memory leaks early in development
2. **Use in CI/CD** - Automate memory leak detection
3. **Profile during development** - Use endpoint profiler to investigate issues
4. **Tune for your tests** - Adjust batch sizes and thresholds in constants.py
5. **Close other apps** - Reduce noise in measurements

## Version

`__version__ = "1.0.0"`
