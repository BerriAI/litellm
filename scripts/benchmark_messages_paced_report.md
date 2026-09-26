# Paced Messages streaming benchmark

The benchmark uses the initial checkout at `5bf13a60ac` with the Anthropic Messages rule set to `RUST_OPT_IN`. It exercises HTTP `/v1/messages` through a single LiteLLM proxy worker, without a database or external callbacks

The mock sends 1,500 timestamped text deltas at 50 deltas/sec over 30 seconds. A delta counts as one synthetic output token; this is not a tokenizer benchmark. Absolute scheduling prevents per-sleep drift from accumulating. The client verifies sequence, count, final usage, stream termination and the `x-litellm-rust` header

Measurements use separate mock, client and proxy processes on the same host. Per-delta delivery latency is measured from the mock's monotonic timestamp to the client's receipt of its SSE data line. Direct-to-mock measurements include the baseline local HTTP/client cost. CPU is user plus system time for the proxy process, including its native threads, divided by completed requests or deltas. It includes request setup and completion work, not just streaming iteration

Concurrency 1 and 20 have three measured rounds, with Python/Rust order reversed in the middle round. Concurrency 100 and 500 have two rounds in opposite orders. Each invocation warms up every connection with five deltas and waits one second. The workload is a synchronized batch of either one or twenty streams, not a saturation test

## Results

Rust did not produce a meaningful throughput gain in this workload. At 500 streams, it used 2.31 times the CPU per completed stream and reached essentially the same peak RSS as Python. Both proxies delivered about 21.3k deltas/sec versus the direct baseline of 24.4k deltas/sec

Measured on an Apple M5 Max with 64 GiB RAM, macOS 26.6.2 and Python 3.12.13. The native extension was built from the pinned worktree with the repository release profile: optimization level 3, fat LTO, one codegen unit. Its SHA-256 is `1c23714abc90c1b21ff2437aed2b09efb66c9d754ac01a954e875b02dbc11618`

Rates, CPU and latency values below are medians across rounds. Peak RSS is the maximum observed across rounds. CPU seconds include all proxy threads; 1.00 core means 100% of one core. All 3,789 measured streams passed validation, covering 5,683,500 text deltas. No swap use was observed in the system snapshots

| Concurrent streams | LITELLM_RUST | Deltas/sec | Requests/sec | CPU cores | CPU ms/request | Peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 50.0 | 0.033 | 0.011 | 333.1 | n/a |
| 1 | 1 | 50.0 | 0.033 | 0.014 | 425.5 | n/a |
| 20 | 0 | 993.8 | 0.663 | 0.090 | 135.5 | 672 |
| 20 | 1 | 992.2 | 0.661 | 0.142 | 215.2 | 619 |
| 100 | 0 | 4,832.3 | 3.222 | 0.261 | 80.9 | 719 |
| 100 | 1 | 4,756.9 | 3.171 | 0.544 | 171.6 | 703 |
| 500 | 0 | 21,261.2 | 14.174 | 0.804 | 56.7 | 1018 |
| 500 | 1 | 21,380.5 | 14.254 | 1.865 | 130.9 | 1018 |

Peak sampling was added after most single-stream trials. Single-stream maximum end RSS was 647 MiB for Python and 594 MiB for Rust

| Concurrent streams | LITELLM_RUST | TTFT p50 ms | Delta delivery p50 ms | Delta delivery p95 ms | Completion p50 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 26.50 | 0.399 | 0.598 | 30.018 |
| 1 | 1 | 26.19 | 0.434 | 0.658 | 30.017 |
| 20 | 0 | 52.07 | 0.961 | 1.640 | 30.166 |
| 20 | 1 | 55.31 | 1.273 | 2.252 | 30.181 |
| 100 | 0 | 182.08 | 0.791 | 2.424 | 30.922 |
| 100 | 1 | 175.27 | 1.929 | 3.413 | 31.118 |
| 500 | 0 | 1131.68 | 1.839 | 209.883 | 34.664 |
| 500 | 1 | 973.12 | 17.696 | 256.221 | 34.561 |

| Direct baseline concurrency | Deltas/sec | TTFT p50 ms | Delta delivery p50 ms | Delta delivery p95 ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 50.0 | 21.80 | 0.163 | 0.264 |
| 20 | 999.0 | 21.17 | 0.163 | 0.403 |
| 100 | 4,969.4 | 21.52 | 0.090 | 0.470 |
| 500 | 24,363.0 | 25.18 | 0.040 | 39.263 |

At one stream, baseline-adjusted median delta delivery cost was about 0.236 ms for Python and 0.271 ms for Rust. TTFT overhead versus direct was about 4.70 ms and 4.39 ms, respectively. Full stream duration remained about 30.02 seconds

At 500 streams, Rust had lower median TTFT but higher median per-delta delivery latency. The direct mock/client baseline already had a roughly 39 ms p95 delivery delay, so high-load tail latency cannot be attributed entirely to either proxy. This is a shared developer workstation with a synchronized, rate-limited workload, two high-load repetitions and no production callbacks or database. These results do not establish maximum throughput or production capacity

Raw per-request JSONL traces and a compact summary were saved locally under `/tmp/litellm-paced-bench/results/` and `/tmp/litellm-paced-bench/summary.json`

## Initial source inspection

The excess CPU is approximately 50 to 60 microseconds per synthetic delta at concurrency 20, 100 and 500. The leading hypothesis is repeated async bridge setup. `PythonDriver::resume_core` boxes a future on each resume and, when pending, allocates abort state and calls `future_into_py`. The pinned `pyo3-async-runtimes` 0.29.0 implementation allocates a Python Future and cancellation channel, spawns two Tokio tasks, and uses a blocking-pool job to attach to Python and schedule completion through `call_soon_threadsafe`. Buffered reads have a ready fast path, so this cost is conditional rather than necessarily paid once for every delta

`MessagesPythonHost::chunk` copies each delivered native buffer into a new Python bytes object. The native Messages relay otherwise forwards raw HTTP bytes without decoding and re-encoding SSE JSON. Both implementations retain delivered chunks for final billing, so that retention is not a Rust-specific explanation for the CPU difference. Request-body clones exist but happen during setup, not once per streamed chunk

These are source-level findings, not sampled CPU attribution. The first optimization experiment should amortize suspended-read bridge setup across a stream, preserving backpressure and cancellation, and compare CPU profiles before attempting zero-copy output

## Reproduction

Use the repository Python environment with `aiohttp`, `psutil`, and the proxy dependencies installed. Build a release native extension and enable the rule before comparing modes

```python
RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN, providers=frozenset({"anthropic"})),
```

Start the standalone mock

```sh
.venv/bin/python scripts/benchmark_messages_paced.py serve --chunks 1500 --rate 50
```

Save this proxy configuration as `/tmp/messages-paced.yaml`

```yaml
model_list:
  - model_name: paced-mock
    litellm_params:
      model: anthropic/claude-sonnet-5
      api_key: fake-provider-key
      api_base: http://127.0.0.1:18098
general_settings:
  master_key: sk-benchmark
litellm_settings:
  telemetry: false
```

Start each proxy in its own shell and record the process PID

```sh
LITELLM_RUST=0 LITELLM_TELEMETRY=False .venv/bin/python litellm/proxy/proxy_cli.py \
  --config /tmp/messages-paced.yaml --host 127.0.0.1 --port 18099
LITELLM_RUST=1 LITELLM_TELEMETRY=False .venv/bin/python litellm/proxy/proxy_cli.py \
  --config /tmp/messages-paced.yaml --host 127.0.0.1 --port 18100
```

Run each command separately for concurrency 1, 20, 100 and 500. Pass the actual proxy PID to include CPU and resident memory measurements. For the reported experiment, run one repeat at a time and alternate mode order across three rounds

```sh
.venv/bin/python scripts/benchmark_messages_paced.py bench --mode direct \
  --url http://127.0.0.1:18098/v1/messages --concurrency 20 --repeats 1 --output direct.jsonl
.venv/bin/python scripts/benchmark_messages_paced.py bench --mode python \
  --url http://127.0.0.1:18099/v1/messages --concurrency 20 --repeats 1 --pid "$PYTHON_PROXY_PID" --output python.jsonl
.venv/bin/python scripts/benchmark_messages_paced.py bench --mode rust \
  --url http://127.0.0.1:18100/v1/messages --concurrency 20 --repeats 1 --pid "$RUST_PROXY_PID" --output rust.jsonl
```

The JSONL output retains each request's TTFT, completion time, per-delta delivery latencies, inter-delta gaps and backend marker. The client rejects truncated, reordered or incomplete streams instead of reporting them as successes
