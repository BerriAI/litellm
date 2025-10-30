# Memory Leak Tests - For Dummies

## What is This?

Tests that check if LiteLLM is leaking memory (eating up more RAM over time). Memory leaks can crash your server.

## Quick Start

```bash
# Run all tests
pytest tests/memory_leak_tests/

# Run specific test
pytest tests/memory_leak_tests/test_sdk_completion_memory.py
pytest tests/memory_leak_tests/test_router_completion_memory.py
```

## What Gets Tested

- **SDK tests**: `litellm.completion()` and `litellm.acompletion()`
- **Router tests**: `Router.completion()` and `Router.acompletion()`
- Both sync/async and streaming/non-streaming modes

## How It Works

1. **Warmup** (3 batches): Let caches stabilize
2. **Measurement** (10 batches): Run requests and measure memory
3. **Analysis**: Check if memory grows or stays stable

Each batch = 100 API requests to a fake LLM endpoint.

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

- `constants.py` - Configuration (batch size, thresholds)
- `memory_test_helpers.py` - Test utilities
- `test_sdk_completion_memory.py` - SDK tests
- `test_router_completion_memory.py` - Router tests

## Adjusting Tests

Edit `constants.py`:

```python
DEFAULT_NUM_BATCHES = 5                              # Run fewer batches (faster)
DEFAULT_BATCH_SIZE = 50                              # Smaller batches (faster)
DEFAULT_LEAK_DETECTION_MAX_GROWTH_PERCENT = 25.0    # Change strictness
```

## Common Issues

**Test takes too long to run**

- Problem: Default runs 10 batches of 100 requests each = 1000 total requests
- Fix: Edit `constants.py` and reduce `DEFAULT_NUM_BATCHES` (try 5) or `DEFAULT_BATCH_SIZE` (try 50)

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
