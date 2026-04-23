# perf harness

Benchmarks LiteLLM + a Rust PyO3 fast-path against a pure-Rust baseline.
Everything self-contained in this directory. Nothing outside is modified.

## Prereqs

```bash
pip install 'litellm[proxy]' yappi httpx maturin
# rust: install via rustup if not present
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# load tool:
brew install wrk         # macOS
# or: git clone https://github.com/wg/wrk && (cd wrk && make) && sudo mv wrk/wrk /usr/local/bin/
```

## Build

```bash
# from this directory
(cd litellm_fast_path && maturin develop --release)   # PyO3 extension
(cd rust_mock_gateway && cargo build --release)       # standalone Rust gateway
```

## Run

Three servers to benchmark. Start one at a time on port 4001 (or `LITELLM_PERF_PORT`).

**1. Plain LiteLLM** (baseline)

```bash
REPO=$(git rev-parse --show-toplevel)
PYTHONPATH=$REPO \
  litellm --config mock_config.yaml --port 4001 --num_workers 4
```

**2. LiteLLM + Rust fast-path middleware** (hybrid)

```bash
REPO=$(git rev-parse --show-toplevel)
PYTHONPATH=$REPO:$(pwd) CONFIG_FILE_PATH=$(pwd)/mock_config.yaml \
  python3 -m uvicorn fast_path_app:app --host 127.0.0.1 --port 4001 --workers 4 --loop uvloop
```

**3. Standalone Rust gateway** (ceiling; uses port 4002)

```bash
PORT=4002 ./rust_mock_gateway/target/release/rust_mock_gateway
```

## Drive load

```bash
# against whichever server is running — swap the port to 4002 for the Rust gateway
wrk -t4 -c50  -d30s --latency -s wrk_post.lua http://127.0.0.1:4001/v1/chat/completions
wrk -t4 -c200 -d30s --latency -s wrk_post.lua http://127.0.0.1:4001/v1/chat/completions
```

Or the orchestrator script (httpx driver, 3 modes, auto-profiles single-worker runs):

```bash
./run_perf.sh single
./run_perf.sh batch
LITELLM_PERF_WORKERS=4 ./run_perf.sh rps 50
LITELLM_PERF_RUNNER=uvicorn+fastpath LITELLM_PERF_WORKERS=4 ./run_perf.sh rps 50
```

Results land in `results/<mode>-<timestamp>/`.

## Expected on an 8-core M-series Mac

`wrk -t4 -d30s -s wrk_post.lua <url>`

| Server | wrk -c | RPS | p50 (ms) | avg (ms) |
|---|---:|---:|---:|---:|
| LiteLLM 4w (plain) | 50 | ~1 100 | ~39 | ~56 |
| LiteLLM 4w + Rust fast-path | 50 | ~27 000 | ~1.4 | ~4.6 |
| Rust axum standalone | 50 | ~77 000 | ~0.45 | ~1.1 |
| Rust axum standalone (peak) | 200 | ~105 000 | ~1.3 | ~4.6 |

p50 is the `Latency Distribution` 50% line in wrk's output.
avg is the `Thread Stats → Latency → Avg` line (includes tail).

If you get numbers in the same ballpark, it's working.

## Troubleshooting

- **port busy**: `LITELLM_PERF_PORT=4005 ...`
- **`litellm_fast_path` import fails**: rerun `maturin develop --release` on your machine; the wheel is CPU+Python-version specific.
- **`py-spy` sudo prompts**: the orchestrator auto-skips profiling when `LITELLM_PERF_WORKERS > 1`.
- **wrk throws `connect`/`read` errors under high concurrency**: that's kernel socket limits, not the server.
