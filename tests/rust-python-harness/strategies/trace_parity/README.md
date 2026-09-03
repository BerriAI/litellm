# Trace Parity Strategy

Gateway endpoint trace validation using `sys.setprofile()` for Python and Rust tracing instrumentation.

## Current Implementation

**Python Tracing:** Uses `sys.setprofile()` with `GatewayProfiler` adapted from `tests/sdk_function_trace/profiler.py`
- Profiles gateway endpoint execution through litellm source
- Captures `(function, depth)` events

**Rust Tracing:** Uses existing `FunctionTrace` infrastructure with `trace=True` parameter
- Rust SDK functions already support `trace=True` via `bridge_route!` macro
- Returns `{"response": ..., "trace": [{"function": ..., "depth": ...}]}`
- Uses `#[tracing::instrument(target = "litellm::function_trace")]` spans

## How It Works

### Python Profiling
```python
with profile_gateway(source_root=litellm_root) as profiler:
    response = client.post("/chat/completions", json=request)
python_trace = tuple(profiler.events)
```

### Rust Trace Collection
```python
rust_result = await achat_completions(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    trace=True
)
rust_trace = tuple(
    FunctionTraceEvent(**e) for e in rust_result["trace"]
)
```

### Comparison
- Validates both traces are non-empty
- Checks critical functions appear in both traces
- Prints side-by-side comparison

## Running Tests

```bash
# Run gateway trace validation
pytest tests/rust-python-harness/strategies/trace_parity/gateway/ -v -s

# Run specific endpoint
pytest tests/rust-python-harness/strategies/trace_parity/gateway/validate_core.py -v -s
```

## Example Output

```
Python trace (25 events):
  proxy/proxy_server.py:123 chat_completion (depth=0)
  main.py:456 completion (depth=1)
  ...

Rust trace (8 events):
  chat_completions (depth=0)
  prepare_chat_completions (depth=1)
  http_request (depth=2)
  ...
```

## Related

- `tests/sdk_function_trace/` - SDK-level trace validation (reference implementation)
- `litellm-rust/crates/python-bridge/src/function_trace.rs` - Rust tracing infrastructure
- `litellm-rust/crates/python-bridge/src/routes/definition.rs` - `bridge_route!` macro with `trace` parameter
