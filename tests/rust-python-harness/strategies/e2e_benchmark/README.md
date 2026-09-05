# End-to-end SDK benchmark

Compare `LITELLM_RUST=0` and `LITELLM_RUST=1` against a separate local HTTP provider process, with no real provider calls, credentials, or Docker required

The initial workload covers synchronous and asynchronous Mistral OCR using the existing `e2e_parity` recording. Other SDK functions are explicitly unimplemented. This measures SDK calls including loopback HTTP transport and response construction. It does not measure gateway overhead, streaming, or concurrent load

## Run

Build the Rust extension in release mode first. An editable `uv sync` normally builds the development profile, which is unsuitable for a Python/Rust performance comparison

```sh
uv sync --frozen --python 3.12
VIRTUAL_ENV="$PWD/.venv" uvx --from maturin==1.15.0 maturin develop --release
uv run --no-sync python -m tests.rust-python-harness run e2e_benchmark \
  --surface sdk --function ocr \
  --benchmark-arg=--output=/tmp/e2e-benchmark.json
```

Keep `--no-sync` on the benchmark command so it uses the extension you just built. Use an otherwise idle machine and run the same command on both revisions when evaluating a change

For a short smoke run:

```sh
uv run --no-sync python -m tests.rust-python-harness run e2e_benchmark \
  --function ocr \
  --benchmark-arg=--profile=small \
  --benchmark-arg=--route=ocr \
  --benchmark-arg=--iterations=10 \
  --benchmark-arg=--warmup=2 \
  --benchmark-arg=--repeats=1 \
  --benchmark-arg=--output=/tmp/e2e-benchmark-smoke.json
```

`run all` also runs this strategy with its defaults. No CI integration is added

## Workloads

The seed cassette stays under `e2e_parity/sdk/ocr/fixtures/data`. The benchmark derives synthetic size variants in memory; it never edits or re-records the correctness fixtures

| Profile | Inline PDF bytes | Response pages |
| --- | ---: | ---: |
| small | 32 KiB | 1 |
| request_medium | 256 KiB | 1 |
| request_large | 2 MiB | 1 |
| response_medium | 32 KiB | 16 |
| response_large | 32 KiB | 128 |

Request variants add PDF comment padding before the final `startxref` marker, preserving existing object offsets and the EOF trailer. The SDK sends base64 plus JSON framing, so wire request sizes exceed the document sizes above. Response variants repeat recorded pages with contiguous indexes and adjusted usage. They exercise realistic response structure, but their page count intentionally varies independently of the input PDF's content

## Measurements

Each backend, route, size, and repeat gets a fresh SDK process for timing and another for memory. Python and Rust execute sequentially, with their order reversed on alternating repeats. The local provider serves preloaded bytes without parsing or capturing request JSON. Its CPU and RSS are outside the SDK measurements

Workers warm up their clients and run an untimed response check before measuring. Python and Rust response digests must match. Every provider request also checks the existing parity harness's User-Agent convention: a Rust run using Python's HTTP path fails instead of reporting a comparison between two Python runs. Missing native extensions, SDK exceptions, timeouts, and incomplete samples fail the run. Workers publish readiness and results atomically in temporary JSON files; stdout and stderr go to a diagnostic log. The controller waits for timing workers to exit and samples RSS only for memory workers. A timeout or interruption terminates and reaps the SDK worker, with bounded shutdown waits for both SDK and provider processes

Latency starts immediately before the SDK call and ends when its result has been returned and discarded. Async calls are awaited on a persistent event loop. CPU is process CPU time during the timed batch, including Python and native threads. Fixture loading, process startup, warmup, preflight serialization, and report generation are excluded. Default SDK behavior is retained, so deferred background work can extend beyond a call's return; these metrics describe the measurement window, not the eventual cost of every callback

The memory controller uses `psutil` to sample only the SDK worker's RSS during a separate run, avoiding polling overhead in latency results. Baseline RSS is taken after warmup and garbage collection. Peak is the highest sampled RSS, including the baseline and final sample. After RSS is measured after the workload and another garbage collection, with input/client state still resident. RSS includes native allocations and shared resident pages, so it is not equivalent to Python heap size or uniquely owned memory. Sampling can miss brief peaks; these values are not an exact allocator high-water mark or proof of a leak

The terminal reports pooled p50/p95/p99 latency, CPU milliseconds per call, sequential calls per second, baseline/peak/after RSS, and speedup (`Python p50 / backend p50`). Throughput is at concurrency one, not saturation capacity. Short runs cannot estimate tail latency reliably. The JSON retains each repeat's raw latency samples, CPU and memory measurements, input/response sizes, seed hash, Python version, native extension hash, settings, platform, Git revision, and whether the working tree has changes

## Options

Pass each option through `--benchmark-arg=...`

| Option | Default | Meaning |
| --- | --- | --- |
| `--iterations=N` | 100 | Measured calls per worker |
| `--warmup=N` | 10 | Warmup calls, followed by one preflight call |
| `--repeats=N` | 3 | Fresh paired runs per workload |
| `--profile=NAME` | All five | Select a size profile; repeat for several |
| `--route=ocr` or `--route=aocr` | Both | Select SDK entrypoint; repeat for both |
| `--timeout=SECONDS` | 120 | Worker readiness and measurement deadline |
| `--sample-interval-ms=N` | 5 | Memory sampling interval, at least 1 ms |
| `--output=PATH` | None | Write a JSON report, including partial results on worker failure |

Run the strategy tests and existing harness checks with:

```sh
uv run --no-sync pytest -o consider_namespace_packages=true \
  tests/rust-python-harness/strategies/e2e_benchmark \
  tests/rust-python-harness/shared tests/rust-python-harness/cli \
  tests/rust-python-harness/strategies/unit_tests_mapping \
  tests/rust-python-harness/strategies/unit_tests_parity \
  tests/rust-python-harness/strategies/unit_tests_rust \
  tests/test_rust_python_harness.py -q
```
