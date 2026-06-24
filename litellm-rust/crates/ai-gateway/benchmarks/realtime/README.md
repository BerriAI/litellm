# Realtime gateway benchmark — pool on/off

Status: **prototype**. The pool is implemented and unit-tested; the results table
below is intentionally left as placeholders pending a live run on a redeploy. Do
not fabricate numbers — a follow-up run fills the cells.

See `../../REALTIME_POOL_DESIGN.md` for the full design and caveats.

## 1. How pooling gets the perf

The prior benchmark (5000 calls / 500 concurrency) showed the gateway adds
**~+115 ms median total** over direct-to-OpenAI, and that the overhead lives
**entirely in session establishment**. Phase decomposition:

| phase            | direct | gateway  | where the cost is                              |
| ---------------- | ------ | -------- | ---------------------------------------------- |
| dial             | ~0     | ~0       | TCP connect to the gateway is local/cheap      |
| **session**      | ~7 ms  | ~358 ms  | gateway dials a fresh upstream WS to OpenAI and waits for `session.created` |
| first audio      | ~0     | ~0       | no added cost                                  |
| streaming        | ~0     | ~0       | no added cost                                  |

So the whole +115 ms is "on each client connect, the gateway opens a fresh
upstream WS to OpenAI and blocks until `session.created` arrives." First-audio and
streaming add nothing.

**The pool removes that handshake from the critical path.** A background task keeps
a few upstream OpenAI sockets already connected and already past `session.created`
(buffered). On a client connect we take a warm socket and relay its buffered
`session.created` to the client via a **local handoff** (sub-ms to low-ms) instead
of dialing OpenAI and waiting. We then splice exactly as before, so first-audio and
streaming are unchanged. The expensive upstream handshake still happens — but
*ahead of time*, off the request's critical path.

On a **pool miss** (burst exceeds warm supply) or a **dead warm socket**, we
fall back to the current fresh-dial path. The pool can only make a connect faster,
never slower or more fragile.

Expected effect: the **session phase** for pooled connects collapses from ~358 ms
toward the local-handoff floor (a few ms), pulling median total back down toward the
direct-OpenAI baseline. Connects that miss the pool still pay the ~358 ms, so the
median improvement tracks the pool hit rate, which is governed by pool size vs.
connect rate (see deploy guidance).

## 2. Steps to reproduce

The harness is the existing Go tool **`ishaan-berri/ws-bench`** (flags
`-host/-key/-insecure/-m/-n/-c/-t/-v`). Both legs use `-n 5000 -c 500 -m
gpt-realtime`.

### Build and run the pooled gateway locally

```bash
cd litellm-rust

# Build the gateway (release for benchmarking).
cargo build --release -p litellm-ai-gateway

# Run it with the pool ON. The env stand-in builds a single OpenAI deployment;
# a real deploy loads model_list from config (see deploy guidance).
export LITELLM_MASTER_KEY=sk-local-master      # gateway bearer key
export OPENAI_API_KEY=sk-...                    # upstream OpenAI key
export OPENAI_REALTIME_MODEL=gpt-realtime       # public alias clients request
export REALTIME_POOL_SIZE=4                      # warm sockets/key; 0 disables
export REALTIME_POOL_MAX_IDLE_SECS=30            # warm-socket lifetime
export HOST=127.0.0.1 PORT=4001

./target/release/litellm-ai-gateway
# logs: "realtime connection pool enabled: target 4 warm sockets/key, max idle 30s"
```

To benchmark **pool OFF** for the same gateway code path, restart with
`REALTIME_POOL_SIZE=0` (the gateway then fresh-dials each connect — the original
behavior):

```bash
REALTIME_POOL_SIZE=0 ./target/release/litellm-ai-gateway
```

### ws-bench: direct-to-OpenAI leg (baseline)

```bash
ws-bench \
  -host api.openai.com \
  -key "$OPENAI_API_KEY" \
  -m gpt-realtime \
  -n 5000 -c 500 -v
```

### ws-bench: gateway leg (run once with pool OFF, once with pool ON)

```bash
ws-bench \
  -host 127.0.0.1:4001 \
  -key "$LITELLM_MASTER_KEY" \
  -insecure \
  -m gpt-realtime \
  -n 5000 -c 500 -v
```

(`-insecure` because the local gateway is plain `ws://`; against a TLS-terminated
deployment, drop `-insecure` and point `-host` at the public hostname.)

Run the gateway leg twice — once with `REALTIME_POOL_SIZE=0`, once with
`REALTIME_POOL_SIZE=4` (or your tuned value) — to fill the OFF and ON columns.

## 3. Deploy guidance

The pooled gateway is a single binary; deploy it like the existing gateway and add
the pool env vars.

**Env vars:**

| env                          | purpose                                                         |
| ---------------------------- | -------------------------------------------------------------- |
| `LITELLM_MASTER_KEY`         | gateway bearer key (required; fails closed if unset)           |
| `OPENAI_API_KEY`             | upstream OpenAI key (env stand-in path)                        |
| `REALTIME_POOL_SIZE`         | target warm sockets per key. `0` disables pooling.             |
| `REALTIME_POOL_MAX_IDLE_SECS`| warm-socket lifetime before it's closed and replaced (default 30) |
| `HOST` / `PORT`              | bind address (default `127.0.0.1:4001`)                        |

**Tuning `REALTIME_POOL_SIZE`:** each warm socket serves exactly one session
(realtime isn't multiplexed), so size to the *connect rate*, not concurrent
connections:

```
REALTIME_POOL_SIZE ≈ peak connect-rate (connects/sec) × warm-socket lifetime (sec)
```

In practice the useful floor is "enough warm sockets to cover the handshake window
at peak connect rate" — start at `4`, watch the pool hit rate, and raise it until
hits plateau. Over-provisioning just burns idle sockets (and idle billing), which
is why warm sockets are short-lived. `0` disables pooling entirely (fresh-dial each
connect) — use it as the control in benchmarks and as a kill switch.

**Interaction with `config.yaml` / render setup:** with the `python-config` feature
and `LITELLM_CONFIG_PATH` set, the gateway loads `model_list` from the proxy
`config.yaml` at startup (load-time only). The pool registers **each** deployment's
upstream `(model, api_key, api_base)` key at startup, so multi-deployment configs
get a warm pool per deployment automatically — no per-model pool config needed. The
pool env vars are orthogonal to `config.yaml`; set them in the same place you set
`LITELLM_MASTER_KEY` (render service env, container env, etc.). The image must run
with the pool: the default `REALTIME_POOL_SIZE=4` means pooling is on unless
explicitly set to `0`.

## Results (5000 calls / 500 concurrency)

Same row/column shape as the prior benchmark. Cells are placeholders pending a live
run on a redeploy.

| metric            | Direct OpenAI         | Gateway pool OFF      | Gateway pool ON       |
| ----------------- | --------------------- | --------------------- | --------------------- |
| success           | TBD — pending live run | TBD — pending live run | TBD — pending live run |
| median dial       | TBD — pending live run | TBD — pending live run | TBD — pending live run |
| median session    | TBD — pending live run | TBD — pending live run | TBD — pending live run |
| median 1st-audio  | TBD — pending live run | TBD — pending live run | TBD — pending live run |
| median total      | TBD — pending live run | TBD — pending live run | TBD — pending live run |
| p95 total         | TBD — pending live run | TBD — pending live run | TBD — pending live run |

Expected shape once filled: **Direct** and **Gateway pool ON** medians close
together (session phase near the local-handoff floor for pool hits); **Gateway pool
OFF** showing the ~+115 ms session-establishment overhead the pool targets.
