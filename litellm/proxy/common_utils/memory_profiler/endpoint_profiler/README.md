# Memory Leak Profiler

Production-ready memory profiling and leak detection system for LiteLLM Proxy endpoints.

## Overview

The Memory Leak Profiler monitors memory usage of proxy endpoints to detect leaks, track growth patterns, and optimize memory consumption. It uses `tracemalloc` for precise memory tracking and implements smart capture strategies to minimize overhead.

## Design Philosophy

1. **Lean & Focused**: Memory profiling only - no CPU/timing bloat
2. **Smart Capture**: Detailed snapshots only when needed (errors, growth, periodic sampling)
3. **Efficient I/O**: Buffered writes to avoid disk I/O on every request
4. **Professional Detection**: Reuses battle-tested algorithms from parent `memory_profiler` module
5. **Production-Ready**: Thread-safe, async-friendly, automatic rotation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    @profile_endpoint()                      │
│                      (Decorator)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               EndpointProfiler (Singleton)                  │
│  • Manages profiling state (enabled/disabled)               │
│  • Coordinates capture & storage                            │
│  • Auto-flush background task                               │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌───────────────────────┐   ┌──────────────────────┐
│  Capture (capture.py) │   │ Storage (storage.py) │
│  • Smart decisions    │   │ • In-memory buffer   │
│  • Tracemalloc usage  │   │ • Async flush        │
│  • Detailed/lite mode │   │ • File rotation      │
└───────────────────────┘   └──────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  JSON Profile Files │
              │  endpoint_profiles/ │
              └──────────┬──────────┘
                         │
                         ▼
              ┌──────────────────────────────────────┐
              │     Analysis (Modular System)        │
              │ • analyze_profiles.py (orchestrator) │
              │ • data_loading.py (load data)        │
              │ • memory_analysis.py (detect leaks)  │
              │ • consumer_analysis.py (consumers)   │
              │ • location_analysis.py (growth)      │
              │ • reporting.py (output)              │
              └──────────────────────────────────────┘
```

## Key Components

### 1. **profiler.py** - Core Orchestrator

- `EndpointProfiler`: Singleton managing profiling lifecycle
- `@profile_endpoint()`: Decorator for automatic profiling
- Auto-flush background task

### 2. **capture.py** - Smart Capture Logic

- **Smart Capture Strategy**: Decides when to capture detailed vs lightweight snapshots
  - Detailed: First request, errors, memory spikes (>2%), periodic (every 25th)
  - Lightweight: All other requests (minimal overhead)
- Uses `tracemalloc` for precise memory tracking
- Reuses parent `memory_profiler.snapshot` module

### 3. **storage.py** - Efficient I/O

- `ProfileBuffer`: Thread-safe in-memory buffer
- Flushes to disk in batches (default: 100 requests or 60 seconds)
- Automatic rotation (keeps last 10,000 profiles per endpoint)
- Async writes to avoid blocking

### 4. **Analysis Modules** - Modular Leak Detection

The analysis system is split into focused modules for better maintainability:

- **analyze_profiles.py**: Main orchestration - coordinates analysis workflow
- **data_loading.py**: Profile data loading from JSON files
- **memory_analysis.py**: Core memory growth and leak detection (uses parent module algorithms)
- **consumer_analysis.py**: Top memory consumers aggregation and reporting
- **location_analysis.py**: Memory growth by code location tracking
- **reporting.py**: Formatted output and report generation

### 5. **utils.py** - Helper Functions

- Memory formatting (`bytes_to_mb`)
- GC forcing (`force_gc`)
- Endpoint name sanitization
- Sampling logic

### 6. **constants.py** - Configuration

- Buffer sizes, flush intervals
- Smart capture thresholds
- File paths and rotation settings

## Modular Analysis System

The analysis subsystem is designed with modularity and single responsibility in mind:

### Module Responsibilities

1. **data_loading.py** - Data Input Layer

   - Load profile data from JSON files
   - Extract and parse request IDs
   - Sort profiles chronologically
   - Pure functions with no side effects

2. **memory_analysis.py** - Core Analysis Engine

   - Extract memory samples from profiles
   - Detect memory leaks using parent module algorithms
   - Calculate rolling averages and growth metrics
   - Returns structured analysis results

3. **consumer_analysis.py** - Resource Consumption Analysis

   - Parse file:line locations from tracemalloc data
   - Aggregate memory usage by location
   - Find top memory-consuming code locations
   - Generate aggregated statistics

4. **location_analysis.py** - Temporal Growth Analysis

   - Track memory usage per location over time
   - Calculate growth metrics (first → last)
   - Identify locations with significant growth
   - Detect code paths with memory accumulation

5. **reporting.py** - Output Formatting Layer

   - Format analysis results for display
   - Print leak detection reports
   - Display growth metrics and recommendations
   - Consistent formatting across all outputs

6. **analyze_profiles.py** - Orchestration Layer
   - Coordinates the analysis workflow
   - Combines results from specialized modules
   - Provides CLI entry point
   - Minimal logic - delegates to specialized modules

### Benefits of Modular Design

- **Single Responsibility**: Each module has one clear purpose
- **Testability**: Easy to unit test individual functions
- **Maintainability**: Changes are isolated to specific modules
- **Reusability**: Functions can be imported independently
- **Readability**: Smaller files are easier to understand
- **Extensibility**: New analysis types can be added as new modules

### Usage Examples

```python
# Use high-level orchestration
from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import analyze_profile_file
results = analyze_profile_file("endpoint_profiles/chat.json")

# Or use individual modules for custom analysis
from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import (
    load_profile_data,
    analyze_endpoint_memory,
    analyze_top_memory_consumers,
)

profiles = load_profile_data("endpoint_profiles/chat.json")
memory_analysis = analyze_endpoint_memory(profiles)
analyze_top_memory_consumers(profiles, top_n=10)
```

## How It Works

### 1. **Initialization** (Server Startup)

```python
from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import EndpointProfiler

profiler = EndpointProfiler.get_instance(
    enabled=True,
    sampling_rate=1.0,      # Profile 100% of requests
    buffer_size=100,        # Flush every 100 profiles
)

await profiler.start_auto_flush()  # Start background flusher
```

### 2. **Request Profiling** (Per Request)

```python
@profile_endpoint()
async def chat_completions(request: Request):
    # Your endpoint logic
    return response
```

**What happens:**

1. Decorator checks if request should be sampled (based on sampling_rate)
2. Captures memory before request
3. Executes endpoint logic
4. Captures memory after request
5. Decides: detailed or lightweight snapshot?
   - **Detailed**: Memory spike, error, first request, or periodic interval
   - **Lightweight**: Just basic stats (memory, latency)
6. Adds profile to buffer (non-blocking)

### 3. **Storage** (Asynchronous)

- Profiles buffer in memory
- Flush triggered by:
  - Buffer full (100 profiles)
  - Time interval (60 seconds)
  - Manual flush
- Writes to: `endpoint_profiles/{endpoint_name}.json`
- Rotates old entries to prevent unbounded growth

### 4. **Analysis** (Post-Collection)

```python
from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import analyze_profile_file

analysis = analyze_profile_file("endpoint_profiles/chat_completions.json")

if analysis.get('leak_detected'):
    print(f"⚠️  Memory leak detected: {analysis['leak_message']}")
    print(f"Growth rate: {analysis['growth_rate_mb_per_request']:.3f} MB/request")
```

## Smart Capture Strategy

Reduces overhead by capturing detailed snapshots only when necessary:

| Trigger        | Condition              | Reason               |
| -------------- | ---------------------- | -------------------- |
| First Request  | `request_counter == 1` | Baseline memory      |
| Error Response | 4xx/5xx status         | Debug errors         |
| Memory Spike   | Memory up >2%          | Detect leaks         |
| Periodic       | Every 25th request     | Regular sampling     |
| **Default**    | **All others**         | **Lightweight only** |

**Lightweight profile** (~50 bytes):

```json
{
  "request_id": "req-123",
  "memory_mb": 245.3,
  "latency_ms": 123.4,
  "status_code": 200
}
```

**Detailed profile** (~2KB with stack traces):

```json
{
  "request_id": "req-1",
  "memory_mb": 245.3,
  "memory_delta_mb": 2.1,
  "top_consumers": [
    {"file": "module.py", "line": 42, "size_mb": 1.2}
  ],
  "gc_stats": {...}
}
```

## Configuration

### Environment Variables

```bash
# Enable/disable profiling
export LITELLM_ENABLE_PROFILING=true

# Sampling rate (0.0 to 1.0)
export LITELLM_PROFILING_SAMPLE_RATE=1.0

# Output directory
export LITELLM_PROFILE_OUTPUT_DIR=endpoint_profiles
```

### Programmatic Configuration

```python
profiler = EndpointProfiler.get_instance(
    enabled=True,                  # Enable profiling
    sampling_rate=0.1,             # Profile 10% of requests
    buffer_size=100,               # Flush every 100 profiles
    flush_interval=60,             # Or every 60 seconds
    max_profiles_per_endpoint=10000,  # Rotation limit
)
```

## Performance Impact

- **Enabled but lightweight capture**: ~0.1ms overhead per request
- **Detailed capture** (smart, infrequent): ~5-10ms overhead
- **Memory overhead**: ~50 bytes per buffered profile
- **Disk I/O**: Async, batched (no blocking)

## Output Files

Profiles are written to `endpoint_profiles/{sanitized_endpoint_name}.json`:

```
endpoint_profiles/
├── chat_completions.json
├── embeddings.json
└── model_info.json
```

Each file contains a JSON array of profiles, automatically rotated to prevent unbounded growth.

## Management API

```python
# Get stats
stats = profiler.get_stats()
# {'requests_profiled': 1523, 'buffer_size': 23, 'enabled': True}

# Enable/disable dynamically
profiler.enable()
profiler.disable()

# Manual flush
profiler.buffer.flush_all()

# Clear buffers
profiler.buffer.clear_all()
```

## Analysis Tools

### Load and Analyze

```python
from litellm.proxy.common_utils.memory_profiler.endpoint_profiler import (
    load_profile_data,
    analyze_endpoint_memory,
)

profiles = load_profile_data("endpoint_profiles/chat_completions.json")
analysis = analyze_endpoint_memory(profiles)

print(f"Requests analyzed: {analysis['request_count']}")
print(f"Memory range: {analysis['memory_min_mb']:.1f} - {analysis['memory_max_mb']:.1f} MB")
print(f"Leak detected: {analysis['leak_detected']}")
```

### Analysis Output

```python
{
    'request_count': 1000,
    'memory_min_mb': 243.1,
    'memory_max_mb': 251.7,
    'memory_mean_mb': 247.4,
    'memory_median_mb': 247.2,
    'leak_detected': False,
    'growth_rate_mb_per_request': 0.008,
    'top_consumers': [...]
}
```

### Command-Line Analysis

Analyze profile files directly from the command line:

```bash
# Basic analysis with default settings (shows top 20 growing locations)
python -m litellm.proxy.common_utils.memory_profiler.endpoint_profiler endpoint_profiles/chat_completions.json

# Show top 30 locations by memory growth
python -m litellm.proxy.common_utils.memory_profiler.endpoint_profiler endpoint_profiles/chat_completions.json --growth 30
```

The command-line tool provides:

- Memory growth analysis
- Leak detection with detailed metrics
- Top memory-consuming locations aggregated across all requests
- Memory growth by location (first vs last appearance)

## Leak Detection Algorithm

Uses professional algorithms from parent `memory_profiler.analysis.detection` module:

1. **Linear Regression**: Fit trend line to memory over requests
2. **Statistical Significance**: R² > 0.7, p-value < 0.05
3. **Growth Rate Threshold**: > 0.01 MB/request
4. **Variance Check**: Exclude high variance (unstable)

## Integration with Parent Memory Profiler Module

The endpoint profiler reuses battle-tested code from parent `memory_profiler` module:

- **Capture logic**: `memory_profiler.snapshot.capture`
- **Analysis algorithms**: `memory_profiler.analysis.detection`
- **Utilities**: `memory_profiler.utils` and `memory_profiler.core`

This ensures consistency between testing and production monitoring.

## Best Practices

1. **Start with sampling**: Use `sampling_rate=0.1` (10%) in high-traffic production
2. **Monitor analysis results**: Check for leaks regularly
3. **Tune smart capture**: Adjust thresholds based on your memory patterns
4. **Use async flush**: Keep `async_flush=True` to avoid blocking
5. **Rotate aggressively**: Lower `max_profiles_per_endpoint` if disk space is limited

## Troubleshooting

**Q: Profiler not capturing anything?**  
A: Check `profiler.enabled` and `sampling_rate`. Verify `tracemalloc.is_tracing()`.

**Q: Too much disk usage?**  
A: Lower `max_profiles_per_endpoint` or increase `sampling_rate` (sample less).

**Q: Performance impact?**  
A: Reduce `sampling_rate` or increase `memory_increase_threshold_pct` for fewer detailed captures.

**Q: Analysis shows no leaks but memory keeps growing?**  
A: Could be gradual growth below detection threshold. Check `growth_rate_mb_per_request`.

## Example Usage

See `__init__.py` for complete usage examples with the proxy server.
