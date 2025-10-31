# Memory Leak Profiler

Production-ready memory profiling and leak detection system for LiteLLM Proxy endpoints.

## Overview

The Memory Leak Profiler monitors memory usage of proxy endpoints to detect leaks, track growth patterns, and optimize memory consumption. It uses `tracemalloc` for precise memory tracking and implements smart capture strategies to minimize overhead.

## Design Philosophy

1. **Lean & Focused**: Memory profiling only - no CPU/timing bloat
2. **Smart Capture**: Detailed snapshots only when needed (errors, growth, periodic sampling)
3. **Efficient I/O**: Buffered writes to avoid disk I/O on every request
4. **Professional Detection**: Reuses battle-tested algorithms from `tests.memory_leak_tests`
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
              ┌────────────────────────┐
              │ Analysis (analyze.py)  │
              │ • Leak detection       │
              │ • Growth analysis      │
              │ • Top consumers        │
              └────────────────────────┘
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
- Reuses `tests.memory_leak_tests.snapshot` when available

### 3. **storage.py** - Efficient I/O

- `ProfileBuffer`: Thread-safe in-memory buffer
- Flushes to disk in batches (default: 100 requests or 60 seconds)
- Automatic rotation (keeps last 10,000 profiles per endpoint)
- Async writes to avoid blocking

### 4. **analyze_profiles.py** - Leak Detection

- Memory growth trend analysis
- Professional leak detection using `tests.memory_leak_tests.analysis`
- Top memory consumers aggregation
- Statistical analysis (mean, median, percentiles)

### 5. **utils.py** - Helper Functions

- Memory formatting (`bytes_to_mb`)
- GC forcing (`force_gc`)
- Endpoint name sanitization
- Sampling logic

### 6. **constants.py** - Configuration

- Buffer sizes, flush intervals
- Smart capture thresholds
- File paths and rotation settings

## How It Works

### 1. **Initialization** (Server Startup)

```python
from tests.memory_leak_tests.profiler import EndpointProfiler

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
from tests.memory_leak_tests.profiler import analyze_profile_file

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
from tests.memory_leak_tests.profiler import (
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
python -m tests.memory_leak_tests.profiler endpoint_profiles/chat_completions.json

# Show top 30 locations by memory growth
python -m tests.memory_leak_tests.profiler endpoint_profiles/chat_completions.json --growth 30
```

The command-line tool provides:

- Memory growth analysis
- Leak detection with detailed metrics
- Top memory-consuming locations aggregated across all requests
- Memory growth by location (first vs last appearance)

## Leak Detection Algorithm

Uses professional algorithms from `tests.memory_leak_tests.analysis.detection`:

1. **Linear Regression**: Fit trend line to memory over requests
2. **Statistical Significance**: R² > 0.7, p-value < 0.05
3. **Growth Rate Threshold**: > 0.01 MB/request
4. **Variance Check**: Exclude high variance (unstable)

## Integration with Memory Leak Tests

The profiler reuses battle-tested code from `tests.memory_leak_tests`:

- **Capture logic**: `tests.memory_leak_tests.snapshot.capture`
- **Analysis algorithms**: `tests.memory_leak_tests.analysis.detection`
- **Utilities**: `tests.memory_leak_tests.utils.formatting`

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
