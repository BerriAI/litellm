# Axum and FastAPI cloud benchmark

Shared tooling for a four-path forwarding comparison. This changes no production routing and includes no completed cloud performance claims

| Row | Request path |
| --- | --- |
| Direct Axum | Client -> Axum -> fixture |
| Direct FastAPI | Client -> FastAPI -> fixture |
| Axum ingress with FastAPI fallback | Client -> Axum -> FastAPI -> fixture |
| FastAPI ingress with Axum fallback | Client -> FastAPI -> Axum -> fixture |

Fallback rows delegate every request before provider execution. The path and a benchmark header select the destination. Authentication, model resolution and production lifecycle behavior are excluded. All paths use the same deterministic native Messages fixture

## Matrix

There is one profile, `litellm-current`. FastAPI uses the repository's `AsyncHTTPHandler`, including its aiohttp transport behind the HTTPX interface. Uvicorn uses uvloop and its automatic HTTP parser selection, matching the proxy CLI. The resolved parser and installed versions are recorded. Python collects request bodies, while both stacks stream responses. Axum forwards request bodies through Hyper without collecting them

The SDK's pool defaults come from repository code, currently 1000 total connections, 500 per host and 15-second upstream keepalive. Rust retains up to 500 idle connections with the same keepalive. Both sides have a one-second connect timeout and 30-second request deadline for the experiment. Each gateway uses one event-loop thread or worker

| Dimension | Values |
| --- | --- |
| Input text | 256 KiB, 1 MiB, 4 MiB, plus a small JSON envelope |
| Response | Complete JSON or streamed SSE |
| Concurrent in-flight requests | 1, 1000, 2000 |
| Measurement | Three 60-second repetitions |
| Warmup | Five seconds per topology |

`matrix.json` defines 18 jobs and 216 measured trials. Each job runs all four topologies sequentially on one host. Order uses a recorded seed. A machine-wide lock prevents concurrent benchmark jobs from different checkouts

Use identical, otherwise idle VM types across workers. Parallelize jobs across machines. At 2000 concurrent 4 MiB requests, each fully buffered copy can occupy roughly 8 GiB before responses and runtime overhead, so provision enough RAM. The common fixture and generator can limit throughput. These tests measure forwarding under controlled load rather than full LiteLLM production capacity

## Setup

Use the repository's Rust toolchain, a C linker, uv, Git, curl and shasum. From the repository root

```bash
uv sync --frozen --extra proxy --group e2e-dev
bash scripts/benchmarks/gateway_split/bootstrap.sh
ulimit -n 32768
```

The existing e2e dependency group supplies the SDK probe client and process sampler. The repository lockfile defines Python dependencies. Bootstrap verifies a pinned Go 1.27.2 archive, checks out Bifrost at `3cb72a95cacaee6f73f9db5636a1a651c7dcb1e9`, applies `bifrost.patch` and builds the binaries. It installs no system services. CGO is disabled to avoid an upstream Apple CPU detector startup crash. Go uses `-trimpath`, and Cargo.lock pins Rust dependencies

## Cloud commands

Regenerate the shared manifest after changing the definition

```bash
.venv/bin/python scripts/benchmarks/gateway_split/cloud.py plan
```

Run one job

```bash
bench=scripts/benchmarks/gateway_split
.venv/bin/python "$bench/cloud.py" run \
  --job litellm-current-4194304-buffered-c1000 \
  --machine-class YOUR_IDENTICAL_VM_CLASS \
  --output "$bench/results/cloud"
```

Or partition across six workers, assigning each a different index from 0 through 5

```bash
bench=scripts/benchmarks/gateway_split
.venv/bin/python "$bench/cloud.py" run \
  --shard-index 0 --shard-count 6 \
  --machine-class YOUR_IDENTICAL_VM_CLASS \
  --output "$bench/results/cloud"
```

Use the same manifest and commit everywhere. Each worker runs its assigned jobs sequentially. Collect the job directories into one location, then merge

```bash
bench=scripts/benchmarks/gateway_split
.venv/bin/python "$bench/cloud.py" merge \
  --inputs "$bench/results/cloud" --output "$bench/results/merged"
```

Completed matching jobs are validated and skipped on rerun. Partial output is rejected, so preserve it for diagnosis and retry in a fresh directory. The merger requires every planned topology/repetition exactly once, passing probes, matching parameters and compatible hardware, dependency lock, versions, source hashes and suite binary

`summary.json` contains four rows per workload with median and range for throughput, completion p50/p95/p99, first-event latency, CPU and sampled RSS. `trials.jsonl` retains individual trials. A median of trial percentiles is not a pooled percentile

A short infrastructure smoke check

```bash
.venv/bin/python scripts/benchmarks/gateway_split/run.py \
  --sizes 4194304 --concurrencies 1 --response buffered \
  --seconds 1 --repetitions 1 --warmup 1
```

## Measurement boundaries

The requested [Bifrost suite](https://github.com/maximhq/bifrost-benchmarking/tree/3cb72a95cacaee6f73f9db5636a1a651c7dcb1e9) drives every trial. Its original concurrency mode timed response headers, closed unread bodies and did not wait for requests launched by workers to finish. The saved patch adds native Messages payload/header files, full-body draining, in-flight completion, latency percentiles, elapsed-time throughput and first complete SSE frame timing

Before loading, live probes verify SDK Messages and SSE reconstruction, request digests, headers, upload forwarding, 429/500 pass-through without retry and upstream cancellation. Every measured request must reach the fixture exactly once and return HTTP 200. SSE fixture completion counts must match client counts, with no unfinished or canceled streams. Request digests are checked during load. Bifrost drains responses, while SDK probes validate returned content before loading

The SSE fixture sends metadata immediately, then eight text deltas with requested five-millisecond sleeps. First-event timing ends at a complete metadata frame and is not time to first text token. Completion timing includes the full response and in-flight drain. Local HTTP/1.1 loopback excludes TLS, provider networking, databases and caches. The fixture parses and hashes every request, which can dominate large-body measurements

Resource samples include both gateways in fallback rows and exclude the fixture and client. RSS is sampled every 100 ms and may double-count shared pages. Source hashes, dependency versions, parser choice and load-generator CPU are recorded

The FastAPI prototype uses the actual SDK transport but omits the proxy's JSON parsing, authentication and logging. Its default transport is aiohttp, while HTTP/2 selects HTTPX in production. The experiment explicitly uses HTTP/1.1. Closed-loop concurrency does not establish externally offered-rate capacity. Bifrost's [run guide](https://docs.getbifrost.ai/benchmarking/run-your-own-benchmarks) distinguishes those two load modes
