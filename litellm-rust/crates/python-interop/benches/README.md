# Byte-stream handoff benchmarks

Run from `litellm-rust`, using an installed Python interpreter with development libraries. No provider, HTTP server, API credentials, or installed LiteLLM extension is needed

```sh
cargo bench -p litellm-python-interop \
  --features pyo3/abi3-py310 --bench streaming
```

Set `PYO3_PYTHON` to choose the interpreter. The executable embeds Python using the production stable ABI target without `extension-module`. It reuses the interpreter, Python event loop, and Tokio runtime

Criterion filters work normally:

```sh
cargo bench -p litellm-python-interop \
  --features pyo3/abi3-py310 --bench streaming \
  -- python_async_for/ready/256
```

Each stream contains 256 chunks of 64, 256, 4096, or 65536 bytes. `ready` yields immediately; `pending` returns `Pending` once before each chunk and wakes its caller. Completion is a no-op future. Payload preparation and stream construction happen outside timing; polling, EOF, completion, and consumed-count validation happen inside

`conversion` compares the production `to_py_bytes` function against the legacy temporary `to_vec()` followed by that same function, with Python already attached. `reader` measures production polling, synchronization, and completion without Python payload objects. `python_async_for` runs the production Python iterator and counts chunks and bytes without retaining payloads, including Python/Tokio scheduling and conversion

The console reports time and byte throughput. The generated `target/criterion/streaming-summary.csv` also reports amortized time per chunk and chunks per second for stream workloads. Conversion rows represent one chunk. Metadata alongside the results records Python version, GIL configuration, Rust compiler and target architecture, ABI selection, arguments, and workload parameters. `CARGO_TARGET_DIR` relocates these outputs

Metadata files have unique timestamped names so baseline and comparison settings remain available. The summary contains only cases measured in the current run; save it before another measurement if needed

Save and compare a baseline using the same interpreter and machine settings:

```sh
cargo bench -p litellm-python-interop --features pyo3/abi3-py310 \
  --bench streaming -- --save-baseline vec
cargo bench -p litellm-python-interop --features pyo3/abi3-py310 \
  --bench streaming -- --baseline vec
```

For the original migration, capture `vec` before replacing the stream's per-chunk `to_vec()` with a moved `Bytes`. The final harness only moves `Bytes`; naming a new baseline `vec` does not restore the old implementation. Only `conversion/legacy_vec` retains that copy for ongoing comparisons

Smoke tests compile and execute all cases, including totals validation, without timing thresholds:

```sh
cargo bench --profile dev -p litellm-python-interop \
  --features pyo3/abi3-py310 --bench streaming -- --test
```

These measurements describe handoff overhead. They do not measure token throughput, provider latency, or allocation counts. Compare stream results directly rather than subtracting isolated layers: scheduling, caching, and allocation behavior interact

Reader byte throughput counts the bytes represented by each handle; the reader does not scan payload contents. It is not a memory-bandwidth measurement
