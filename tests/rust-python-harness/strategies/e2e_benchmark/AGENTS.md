# What this is

Measures Python and Rust SDK latency, process CPU time, and sampled RSS against local provider replays. Initial coverage is sync/async Mistral OCR at concurrency one. Other SDK functions remain explicitly unimplemented. Run locally, with no CI integration

# How it works

Derive five profiles from the existing `e2e_parity` recording without editing it. `small` uses a 32 KiB PDF and one response page. `request_medium` and `request_large` increase only the PDF to 256 KiB and 2 MiB. `response_medium` and `response_large` increase only the response to 16 and 128 pages. Insert PDF comment padding before the original final `startxref` and EOF trailer, preserving object offsets. Response pages are synthetic repetitions, independent of actual PDF content

Generate fixtures in the controller, outside SDK worker imports. Each backend, route, profile, and repeat gets fresh timing and memory workers against a separate local HTTP provider process. Run Python and Rust sequentially, reversing their order on alternating repeats. The provider drains request bytes and serves preloaded responses without JSON parsing or capture. Its CPU and RSS are excluded

Warmup, preflight response checks, bounded-memory native hashing, and garbage collection precede readiness. Require matching Python/Rust preflight digests and matching timing/memory worker digests. Every request checks the parity harness's User-Agent convention to reject Python fallback during a Rust run. Use `e2e_parity` for complete request and response semantics

Time each SDK call until its returned result is discarded. Async calls share a persistent event loop. Process CPU and batch elapsed time include sample collection overhead and work on Python/native threads. Startup, fixture loading, warmup, preflight, final garbage collection, and reporting are excluded. Deferred callbacks can outlive the measured batch

Sample only the SDK worker's RSS in the separate memory pass. Baseline follows warmup and garbage collection. Peak includes baseline, periodic samples, and the final sample. After RSS follows another garbage collection with input/client state resident. RSS includes native allocations and shared pages. Sampling can miss short peaks, and retained RSS does not prove a leak

Publish readiness/results atomically in temporary JSON files, reserving stdout/stderr for diagnostics. Bound readiness, measurement, and shutdown waits. Terminate and reap workers on failure or interruption. Reject missing extensions, backend mismatches, exceptions, and incomplete samples. Preserve completed pairs in partial reports when a worker fails

Report pooled p50/p95/p99 latency, CPU milliseconds per call, sequential calls per second, baseline/peak/after RSS, and Python p50 divided by backend p50. JSON retains raw per-repeat samples, fixture/extension hashes, Python version, options, platform, Git revision, and working-tree state. A pass confirms valid measurements without imposing performance thresholds. Use an idle host, inspect repeat variation, and treat short-run tail estimates cautiously. Loopback HTTP, allocator behavior, and deferred work affect results. Streaming, gateway overhead, live provider latency, and concurrent throughput are outside scope

Editable `uv sync` builds a development extension. Build release explicitly and retain `--no-sync`:

```sh
uv sync --frozen --python 3.12
VIRTUAL_ENV="$PWD/.venv" uvx --from maturin==1.15.0 maturin develop --release
uv run --no-sync python -m tests.rust-python-harness run e2e_benchmark \
  --surface sdk --function ocr \
  --benchmark-arg=--output=/tmp/e2e-benchmark.json
```

Forward each option through `--benchmark-arg=...`. Defaults: `--iterations=100`, `--warmup=10`, `--repeats=3`, `--timeout=120` seconds, `--sample-interval-ms=5`. Repeat `--profile=NAME` or `--route=ocr|aocr` to select subsets. `--output=PATH` exports JSON. Invalid values and unwritable destinations produce handled CLI errors. `run all` includes this strategy. A smoke run can select `small`, `ocr`, 10 iterations, 2 warmups, and 1 repeat

Run focused checks with `uv run --no-sync pytest -o consider_namespace_packages=true tests/rust-python-harness/strategies/e2e_benchmark tests/rust-python-harness/cli -q`
