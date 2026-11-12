# Rust vs Python proxy benchmark

A/B driver that runs the proxy from several commits against the same frozen traffic and reports per-request CPU, gateway overhead and saturation throughput with bootstrap confidence intervals. The variants and the reasoning behind them are in `variants.yaml`. It is a standalone script rather than a pytest perf case because every variant runs from its own git worktree and venv

The harness side (mocks in `tests/e2e/perf/setup/mocks/`, the Locust runner `tests/e2e/perf/setup/locustfile.py` and the corpus) always comes from this checkout, so only the proxy changes between variants. Keep it that way. A variant that runs its own copy of the harness measures harness drift too

## Host

Run on a dedicated Linux host with the CPU governor set to `performance`, nothing else running, and enough cores to pin the proxy, the load generator and the mock services apart (`--proxy-cores`, `--loadgen-cores`, `--service-cores`, `--workers`). The defaults fit a 4 vCPU box (2 proxy workers on cores 0 and 1, Locust on 2, mock and S3 server on 3). On a 16 vCPU host give the proxy 4 cores and 4 workers and the load generator 4 cores. Never compare runs taken with different core layouts or on different hosts

No Postgres or Redis. S3 logging goes to a local `moto_server` and Prometheus uses a fresh `PROMETHEUS_MULTIPROC_DIR` per run

## Steps

```bash
scripts/bench_rust_vs_python/setup_worktrees.sh ~/bench/wt base=d06b889d41 head=2c7ed0713a router=85bc58085e
```

That builds `_native` in release mode with frame pointers, line tables and no symbol stripping in every tree, baseline included, so profiles show Rust frames and every variant pays the same frame-pointer cost. The proxy venvs are synced with `--extra proxy` from each tree's own `uv.lock`, plus `prometheus-client` and `boto3` at the locked versions. Create one separate harness venv with `locust`, `fastapi`, `uvicorn`, `pydantic`, `boto3`, `moto[server]`, `py-spy` and `pyyaml` and point `BENCH_HARNESS_PYTHON` at its python

```bash
python scripts/bench_rust_vs_python/gen_requests.py --path messages --seed 101 --rps 10 --seconds 180 --out ~/bench/plans/l1/messages.json
```

```bash
python scripts/bench_rust_vs_python/run_bench.py --worktrees ~/bench/wt --plans ~/bench/plans/l1 --out ~/bench/results/l1 --variants A0,A0b,A1,B0,B1 --reps 5
```

```bash
python scripts/bench_rust_vs_python/report.py ~/bench/results/l1 --pairs A0:A1,B0:B1 --noise-pair A0:A0b
```

`--ramp first,last,step,step_seconds` on `gen_requests.py` gives the stepped plan for saturation, and `--input-tokens low,high` replays the same arrivals with other request sizes. `--profile-seconds` on `run_bench.py` attaches `py-spy --native` once warmup is over, at `--profile-rate` samples a second, and `flame.py <old run dir> <new run dir> --out <dir>` turns two profiled runs into a subsystem table, a flame tree and a per-function comparison, all in CPU ms per request

Profile at low load (3 RPS) and 50 Hz. At 200 Hz on 4 vCPU py-spy fell minutes behind. Even when it keeps up, attaching it makes the proxy burn several times its normal CPU, so profiled runs are never measured runs, and `flame.py` scales each profile to the CPU per request the same run used before the profiler attached. `--native` counts parked threads as running, so `flame.py` drops samples in thread waits, GIL reacquisition and uvicorn's supervisor pipe read. Every worker's stack starts in uvicorn's supervisor module, so match only the pipe read there, or the whole request path disappears. The profiler slows the Rust variant more than the Python one (more threads to stop), so compare shapes and shares more than exact milliseconds

`litellm-rust/crates/python-bridge/benches/messages_body.rs` times each Python and Rust conversion of a coding-agent request at about 2k and 50k tokens (`PYO3_PYTHON=python3.12 cargo bench -p litellm-python-bridge --bench messages_body`)

## Validity checks

A run counts only when it took the intended path. Rust streams carry `x-litellm-rust: true` (the `rust header share` column). Non-streaming Rust messages responses are plain dicts and carry no header, and the upstream user agent is `litellm/<version>` on both paths, so it proves nothing about the path either. On main, native anthropic `/v1/messages` is Rust-only and ignores `LITELLM_RUST`, so there is no Rust-off control for it on the same commit, and the Python handler for it only refuses, which makes every 200 from that deployment a Rust response. The S3 object count and the Prometheus `litellm_requests_metric` delta must both cover every successful request, and Locust must stay under 70% of its core. Settings that put a messages request back on Python (retries, fallbacks, response caching, prompt-caching injection, anything in `MESSAGES_PYTHON_FEATURES` in `litellm/rust_bridge/catalog.py`) must stay out of `configs/bench.yaml`

Run an A/A pair first. A difference smaller than the A/A interval is noise, whatever its own CI says
