# Benchmarking an ai-gateway endpoint

A reusable method for measuring what the LiteLLM Rust ai-gateway *adds* on top of
talking to a provider directly — for **any** endpoint (realtime WebSocket, chat
completions, responses, embeddings, …). The point of a gateway benchmark is never
an absolute number; it's the **delta vs. provider-direct**, decomposed by phase, at
realistic scale.

> **Hard rule — never commit secrets.** API keys (provider keys, gateway master
> keys) are passed via env vars or CLI flags **only**. They never appear in a
> committed file, a Dockerfile, a results table, a log you paste, or this skill.
> Load generators must read the key from a flag (`-key`) or `$OPENAI_API_KEY` /
> equivalent — never hardcode it. Scrub every harness file before committing.

## The method

1. **Deploy the gateway** under test the way it will actually run (same image,
   same instance plan, same instance count). Don't benchmark a debug build or a
   single instance if production is autoscaled — the contention profile differs.
   Wait until every instance reports healthy (`/health/readiness` → 200).

2. **Use one load generator that hits BOTH legs with the same protocol.** The same
   binary, same flags (`-n` total calls, `-c` concurrency, `-t` timeout), pointed
   once at the **provider directly** and once at the **gateway**. Same model, same
   payload. If the two legs use different clients you're comparing clients, not the
   gateway.

3. **Phase-decompose the latency.** A single "total" number hides where the cost
   lives. Break each call into the protocol's natural phases and time each one
   independently. For the realtime WS endpoint that's:

   | phase         | meaning                                                    |
   | ------------- | ---------------------------------------------------------- |
   | dial          | TCP + TLS + HTTP→WS upgrade handshake                      |
   | session       | upgrade done → first server "ready" event (`session.created`) |
   | first-token   | request sent → first streamed delta (`…audio.delta`)      |
   | total         | full wall-clock per call                                   |

   For a chat/responses endpoint the analogous phases are dial → TTFB (first SSE
   chunk) → last-token → total. The principle is the same: find the phase the
   gateway touches and isolate it, so a regression shows up in *one* column.

4. **Compare at scale.** Run enough calls at enough concurrency to expose queueing,
   pool exhaustion, and tail behavior — not just a warm single call. Run each leg
   **at least twice** and keep the stable run; a cold first run (DNS, TLS session
   cache, JIT pool warm-up) is not representative.

5. **Report success% + p50 + p95 per phase, per leg.** Success rate first (a fast
   benchmark that drops 5% of calls is not fast). Then medians to show the typical
   case and p95 to show the tail. Always state the load (`N calls / C concurrency`)
   and the instance count next to the table.

## Layout convention

One directory per endpoint under `benchmarks/`:

```
benchmarks/
  SKILL.md              ← this file (generic method)
  <endpoint>/           ← one per endpoint, e.g. realtime/
    main.go (or .py …)  ← the load generator (no hardcoded keys)
    Dockerfile          ← builds the generator into a runnable image
    run.sh              ← entrypoint wrapper (decodes flags, execs the binary)
    go.mod / go.sum     ← (or requirements.txt, etc.)
```

Keep the harness self-contained and key-free. The results themselves live in the
**PR description** (or a benchmark log), not in a committed README — numbers go
stale and the repo shouldn't carry a snapshot that drifts from reality.

## Running from a hosted runner (recommended at scale)

Generating hundreds of concurrent connections from a laptop understates gateway
performance — a starved generator (NIC, CPU, ephemeral-port exhaustion) shows up as
*the gateway* being slow. Run the load generator from a **multi-CPU** host close to
both the provider and the gateway:

- Build the `<endpoint>/` harness into an image; deploy it as a long-lived service
  that idles (`sleep infinity`).
- Fire each leg as a **one-off job** against that service and capture stdout from
  the job's logs.
- Size the runner generously (e.g. 8 vCPU for ~500 concurrent WS dials). Re-run any
  leg whose success rate or dial times look generator-bound, on a bigger box.

## Example: realtime 5k / 500

The `realtime/` harness measures the realtime WebSocket endpoint. The gateway's
only added cost is the **session** phase: on each client connect the gateway opens
a fresh upstream WS to the provider and blocks until `session.created`. A
**pre-warmed connection pool** (`REALTIME_POOL_SIZE`) moves that handshake off the
request's critical path — a background task keeps warm upstream sockets already past
`session.created`, and a client connect takes a warm socket via a local handoff
instead of dialing the provider.

**Pool sizing.** Each warm socket serves exactly one session (realtime isn't
multiplexed), so size to the *peak concurrent connects per instance*, not to total
load:

```
REALTIME_POOL_SIZE ≈ peak concurrency ÷ instance count
```

At 5000 calls / 500 concurrency over 10 instances that's ≈ 50 warm sockets per
instance. Over-provisioning just burns idle upstream sockets (and idle billing),
which is why warm sockets are short-lived (`REALTIME_POOL_MAX_IDLE_SECS`, default
30s). Pre-warming opens that many idle provider sockets per instance at boot — if
the provider rejects the warm-up at that level, dial the pool size down. Set
`REALTIME_POOL_SIZE=0` to disable pooling entirely (fresh-dial each connect); use it
as the control/baseline column in the benchmark and as a kill switch.

Run all three legs with `-m gpt-realtime -n 5000 -c 500 -t 60`, twice each:

```bash
# Direct provider (baseline) — key via flag, never hardcoded
wsbench -host api.openai.com -key "$OPENAI_API_KEY" -m gpt-realtime -n 5000 -c 500 -t 60

# Gateway, pool ON (REALTIME_POOL_SIZE≈50) and pool OFF (REALTIME_POOL_SIZE=0)
wsbench -host <gateway-host> -key "$LITELLM_MASTER_KEY" -m gpt-realtime -n 5000 -c 500 -t 60
```

Report **Direct · pool OFF · pool ON** columns for success / median dial / median
session / median first-audio / median total / p95 total. The expected shape:
**pool OFF** carries the gateway's session-establishment overhead; **pool ON** pulls
the session phase down toward the local-handoff floor, bringing median total back
toward the direct baseline. The measured table for a given run lives in that run's
PR description.
