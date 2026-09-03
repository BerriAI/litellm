# Trace Parity Strategy

Gateway endpoint trace validation following the `sdk_function_trace` pattern.

## Current Status

**Phase 1 (Current):** Python execution profiling using `sys.setprofile()`
- Uses `GatewayProfiler` adapted from `tests/sdk_function_trace/profiler.py`
- Profiles gateway endpoint execution through litellm source
- Captures (function, depth) events
- Validates HTTP response success

**Phase 2 (TODO):** Rust trace collection
- Need Rust gateway endpoints to support `trace=True` parameter
- Should return `{"response": ..., "trace": [{"function": ..., "depth": ...}]}`
- Requires `#[tracing::instrument]` on gateway endpoint functions

**Phase 3 (TODO):** Parity assertion
- Port `assert_function_trace_parity` pattern from `sdk_function_trace/harness.py`
- Compare Python trace vs Rust trace
- Fail on missing, extra, or reordered steps
- Verify identical (function, depth) sequences

## How It Works

### Python Profiling
```python
with profile_gateway(source_root=litellm_root) as profiler:
    response = client.post("/chat/completions", json=request)
python_trace = tuple(profiler.events)
```

### Expected Rust Pattern (TODO)
```python
rust_response = rust_client.post("/chat/completions", json=request, params={"trace": "true"})
rust_trace = tuple(FunctionTraceEvent(**e) for e in rust_response.json()["trace"])
```

### Parity Check (TODO)
```python
if python_trace != rust_trace:
    raise AssertionError(f"Traces differ: {python_trace!r} != {rust_trace!r}")
```

## Differences from sdk_function_trace

- **Scope:** Gateway HTTP endpoints, not SDK functions
- **Entry:** `/chat/completions`, `/ocr` endpoints vs `litellm.completion()`, `litellm.ocr()` functions
- **Fixture:** TestClient POST requests vs direct SDK calls
- **Source:** `litellm/proxy/` execution vs `litellm/` SDK execution

## Running Tests

```bash
# Run gateway trace validation
pytest tests/rust-python-harness/strategies/trace_parity/gateway/

# Run specific endpoint
pytest tests/rust-python-harness/strategies/trace_parity/gateway/validate_core.py
```

## Related

- `tests/sdk_function_trace/` - SDK-level trace validation (reference implementation)
- `tests/sdk_function_trace/profiler.py` - Python profiler pattern
- `tests/sdk_function_trace/harness.py` - `assert_function_trace_parity` pattern
