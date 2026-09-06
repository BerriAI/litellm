# What this is

Measures Python and Rust SDK latency, process CPU time, and sampled RSS against local provider replays. Initial coverage is sync/async Mistral OCR at concurrency one. Other SDK functions remain explicitly unimplemented. Keep implementation in this strategy folder. The local CLI reports latency, CPU, and RSS; the CodSpeed adapter uploads walltime measurements from the existing repository workflow

# How it works

Derive five profiles from the existing `e2e_parity` recording without editing it. `small` uses a 32 KiB PDF and one response page. `request_medium` and `request_large` increase only the PDF to 256 KiB and 2 MiB. `response_medium` and `response_large` increase only the response to 16 and 128 pages. Insert PDF comment padding before the original final `startxref` and EOF trailer, preserving object offsets. Response pages are synthetic repetitions, independent of actual PDF content

Generate fixtures in the controller, outside SDK worker imports. Each backend, route, profile, and repeat gets fresh timing and memory workers against a separate local HTTP provider process. Run Python and Rust sequentially, reversing their order on alternating repeats. The default four repeats balance backend order. Timing workers run at least the requested iteration count and minimum duration; the separate memory workers use the same fixed iteration count for both backends The provider drains request bytes and serves preloaded responses without JSON parsing or capture. Its CPU and RSS are excluded

Warmup, preflight response checks, bounded-memory native hashing, and garbage collection precede readiness. Require matching Python/Rust preflight digests and matching timing/memory worker digests. Every request checks the parity harness's User-Agent convention to reject Python fallback during a Rust run. Use `e2e_parity` for complete request and response semantics

Time each SDK call until its returned result is discarded. Async calls share a persistent event loop. Process CPU and batch elapsed time include sample collection overhead and work on Python/native threads. Startup, fixture loading, warmup, preflight, final garbage collection, and reporting are excluded. Deferred callbacks can outlive the measured batch

Sample only the SDK worker's RSS in the separate memory pass. Baseline follows warmup and garbage collection. Peak includes baseline, periodic samples, and the final sample. After RSS follows another garbage collection with input/client state resident. RSS includes native allocations and shared pages. Sampling can miss short peaks, and retained RSS does not prove a leak

Publish readiness/results atomically in temporary JSON files, reserving stdout/stderr for diagnostics. Bound readiness, measurement, and shutdown waits. Terminate and reap workers on failure or interruption. Reject missing extensions, backend mismatches, exceptions, and incomplete samples. Preserve completed pairs in partial reports when a worker fails

Report pooled p50/p95/p99 and mean/standard-deviation latency, per-repeat sample counts, elapsed time, medians and throughput, CPU milliseconds per call, sequential calls per second, baseline/peak/after RSS, and Python p50 divided by backend p50. JSON retains raw per-repeat samples, fixture/extension hashes, Python version, options, platform, Git revision, and working-tree state. JSON schema version 2 also exports diagnostic warnings. A pass confirms completed measurements without establishing statistical significance or imposing performance thresholds. Warn for single repeats, unbalanced order, batches shorter than one second, or a repeat-median range exceeding 10% of its median. These are diagnostic heuristics; absence of warnings is not proof of stability. Pooled statistics weight each call equally, so duration-based sampling can weight faster repeats more heavily. Inspect independent repeat statistics before interpreting pooled ratios. Calls per second follows batch mean time, not median latency. Keep slow calls in the data Use an idle host, inspect repeat variation, and treat short-run tail estimates cautiously. Loopback HTTP, allocator behavior, and deferred work affect results. Streaming, gateway overhead, live provider latency, and concurrent throughput are outside scope

Editable `uv sync` builds a development extension. Build release explicitly and retain `--no-sync`:

```sh
uv sync --frozen --python 3.12
VIRTUAL_ENV="$PWD/.venv" uvx --from maturin==1.15.0 maturin develop --release
uv run --no-sync python -m tests.rust-python-harness run e2e_benchmark \
  --surface sdk --function ocr \
  --benchmark-arg=--output=/tmp/e2e-benchmark.json
```

Forward each option through `--benchmark-arg=...`. Defaults: `--iterations=100`, `--warmup=10`, `--repeats=4`, `--min-time=1` second, `--timeout=120` seconds, `--sample-interval-ms=5`. Repeat `--profile=NAME` or `--route=ocr|aocr` to select subsets. `--output=PATH` exports JSON. Invalid values and unwritable destinations produce handled CLI errors. `run all` includes this strategy. A smoke run can select `small`, `ocr`, 10 iterations, 2 warmups, 1 repeat, and `--min-time=0`

Run focused checks with `uv run --no-sync pytest -o consider_namespace_packages=true tests/rust-python-harness/strategies/e2e_benchmark tests/rust-python-harness/cli -q`

The CodSpeed job in `.github/workflows/codspeed.yml` uses `mode: walltime` on `codspeed-macro`, with release compilation outside instrumentation. It leaves the existing CPU simulation job intact. Macro runners need explicit access to this public repository through the organization runner group. The ARM64 release cache includes runner architecture

Run the CodSpeed adapter in place:

```sh
uv pip install --python .venv/bin/python pytest-codspeed==5.0.3
uv run --no-sync python -m tests.rust-python-harness.strategies.e2e_benchmark.codspeed
```

Use `--profile=small`, `--route=ocr|aocr`, and `--max-time=5` seconds to select work. Each backend/profile/route runs once in a fresh pytest process with a stable CodSpeed benchmark ID. Setup, provider startup, preflight validation, and teardown are outside the benchmark fixture. Compare Python/Rust response digests after both processes finish. The same provider rejects backend fallback. Timed warmup and round-count selection are supplied by CodSpeed, which stores history and profiling data in CI. Repeated rounds in one process do not replace independent process repeats; use the local CLI for that audit

Set CodSpeed `min_time=0` so each round measures one full SDK call. This also avoids the pinned plugin's terminal display dividing normalized timings by iterations a second time for multi-call rounds. Async measurements await completion on a persistent loop, including `run_until_complete` entry/exit per call; local CLI async samples run inside the loop, so compare backends within each instrument rather than equating their absolute timings. Neither path overrides the logging executor. CodSpeed's best-time statistic is not the CLI's median or throughput. Memory and process CPU remain separate local measurements

Interpret results per route and profile. A synchronous small-request improvement does not establish an async improvement or monotonic scaling. Neither path isolates PyO3 overhead. Sampled peak RSS reductions do not imply equivalent retained-memory reductions. Use a separate bridge microbenchmark and allocation profiling to investigate causes

Methodology references: [Switowski](https://switowski.com/blog/how-to-benchmark-python-code/) on repeatability and setup boundaries, [CodSpeed](https://codspeed.io/docs/instruments/walltime) on I/O-inclusive measurement, [UCL](https://github-pages.arc.ucl.ac.uk/python-tooling/pages/benchmarking-profiling.html) on benchmarking versus profiling, and [Bencher](https://bencher.dev/learn/benchmarking/python/pytest-benchmark/) on distributions and mean-based throughput
