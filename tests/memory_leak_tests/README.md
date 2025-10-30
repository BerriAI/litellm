# Memory Leak Tests - For Dummies

## What is This?

Tests that check if LiteLLM is leaking memory (eating up more RAM over time). Memory leaks can crash your server.

## Quick Start

```bash
# Run all tests
pytest tests/memory_leak_tests/

# Run specific test
pytest tests/memory_leak_tests/tests/test_sdk_completion.py
pytest tests/memory_leak_tests/tests/test_router_completion.py
```

## What Gets Tested

- **SDK tests**: `litellm.completion()` and `litellm.acompletion()`
- **Router tests**: `Router.completion()` and `Router.acompletion()`
- Both sync/async and streaming/non-streaming modes

## How It Works

1. **Warmup** (2 batches): Let caches stabilize
2. **Measurement** (6 batches): Run requests and measure memory
3. **Analysis**: Check if memory grows or stays stable

Each batch = 50 API requests to a fake LLM endpoint.

**Performance**: Each test now runs **400 total requests** (optimized from 1,300) with smart snapshot capture to balance speed and accuracy.

## Reading Results

**Good (No Leak)**

```
Memory Growth Analysis
Initial avg: 15.23 MB
Final avg:   15.45 MB
Growth:      0.22 MB (1.4%)

Memory stabilized — no leak detected
```

**Bad (Leak Detected)**

```
Memory Growth Analysis
Initial avg: 15.23 MB
Final avg:   20.87 MB
Growth:      5.64 MB (37.0%)

Memory grew by 37.0% (>25% threshold) — possible leak
```

**Error-Induced Leak**

```
ERROR-INDUCED MEMORY LEAK DETECTED

Memory spikes occurred in batch(es) with errors and did not fully recover:
  • Batch 5: 15.23 MB → 23.45 MB (+54.0%) with 12 error(s) → stabilized at 23.50 MB

This indicates that error handling is not properly releasing resources.
```

## Files

The codebase is organized into modular components:

### Core Modules

- `constants.py` - Configuration (batch size, thresholds, smart capture settings)
- `core/` - Core execution logic
  - `execution.py` - Main test execution engine
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

### Test Suites

- `tests/` - Actual test files
  - `test_sdk_completion.py` - SDK completion tests
  - `test_router_completion.py` - Router completion tests

### Utilities

- `utils/` - Helper functions
  - `conversions.py` - Data type conversions
  - `formatting.py` - Output formatting

## Adjusting Tests

Edit `constants.py`:

```python
# Execution speed controls (current optimized values)
DEFAULT_NUM_BATCHES = 6                              # Number of measurement batches
DEFAULT_BATCH_SIZE = 50                              # Requests per batch
DEFAULT_WARMUP_BATCHES = 2                           # Warmup batches before measurement
DEFAULT_LEAK_DETECTION_MAX_GROWTH_PERCENT = 25.0    # Leak detection threshold

# Smart capture configuration (controls snapshot frequency)
DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT = 2.0          # Capture when memory grows by >2%
DEFAULT_PERIODIC_SAMPLE_INTERVAL = 25                # Capture every 25 requests
DEFAULT_TOP_CONSUMERS_COUNT = 10                     # Top memory consumers to track
```

**For even faster tests** (less confidence):

```python
DEFAULT_NUM_BATCHES = 4
DEFAULT_BATCH_SIZE = 30
DEFAULT_WARMUP_BATCHES = 1
DEFAULT_PERIODIC_SAMPLE_INTERVAL = 50
```

## Smart Capture Configuration

Tests capture detailed memory snapshots (including top memory consumers) selectively to balance detail with performance:

**When detailed snapshots are captured:**

1. **First request** - Always (baseline)
2. **On errors** - Always (to debug error-related leaks)
3. **Memory increases** - When memory grows by more than `DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT` (default 2%)
4. **Periodic sampling** - Every `DEFAULT_PERIODIC_SAMPLE_INTERVAL` requests (default 100)

**Between these events**, lightweight snapshots are captured (basic metrics only, no expensive profiling).

**Tuning Smart Capture:**

```python
# More sensitive (catches smaller leaks, more snapshots, larger files)
DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT = 1.0
DEFAULT_PERIODIC_SAMPLE_INTERVAL = 50

# Less sensitive (fewer snapshots, smaller files, may miss small leaks)
DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT = 5.0
DEFAULT_PERIODIC_SAMPLE_INTERVAL = 200
```

**When to adjust:**

- **Lower threshold**: Testing code prone to small, gradual leaks (more sensitive detection)
- **Higher threshold**: Long-running tests where file size is a concern (faster, less data)
- **Shorter interval**: Critical paths where you want comprehensive coverage (slower, more snapshots)
- **Longer interval**: Stable code where periodic verification is sufficient (faster, recommended)

**Performance Impact:**

- `PERIODIC_SAMPLE_INTERVAL = 10`: ~130 detailed snapshots per test (slower, comprehensive)
- `PERIODIC_SAMPLE_INTERVAL = 25`: ~16 detailed snapshots per test (current, balanced)
- `PERIODIC_SAMPLE_INTERVAL = 50`: ~8 detailed snapshots per test (faster, less detail)

## Performance Optimizations

The tests are optimized for speed while maintaining leak detection accuracy:

- **400 requests per test** (2 warmup + 6 measurement batches × 50 requests)
- **Smart snapshot capture** - detailed profiling only every 25 requests (not every request)
- **Reduced profiling overhead** - tracks top 10 consumers (not 20)
- **~69% faster** than previous configuration

**Approximate test duration:**

- Single test variant: ~30-60 seconds
- All 4 variants (per test file): ~2-4 minutes
- Full test suite: ~5-8 minutes

## Common Issues

**Test takes too long to run**

- Current: 400 requests per test (already optimized)
- Further reduction: Edit `constants.py` and reduce `DEFAULT_NUM_BATCHES` (try 4) or `DEFAULT_BATCH_SIZE` (try 30)
- Warning: Too few requests may miss slow memory leaks

**Test is flaky or skips with "too noisy" message**

- Problem: Other processes on your machine are using memory, making measurements unreliable
- Fix: Close other applications or run on a quieter machine
- Note: Test auto-skips if measurements vary by more than 40%. To adjust, change `DEFAULT_MAX_COEFFICIENT_VARIATION` in `constants.py`

**Need to test against your own endpoint**

- Problem: Tests use a fake endpoint by default
- Fix: Edit `FAKE_LLM_ENDPOINT` in `constants.py` to point to your server

## Debugging Tips

If a test fails, check:

1. Unreleased resources (HTTP connections, file handles)
2. Error handling cleanup
3. Callbacks creating closures with large objects
4. Unbounded caches

## Key Terms

- **Batch**: Group of API requests
- **Warmup**: Initial batches to stabilize before measurement
- **Rolling Average**: Smooths measurement noise
- **Growth Percentage**: Memory increase from start to end
