# Realtime route (`GET /v1/realtime`)

Proxies OpenAI's realtime WebSocket. `mod.rs` is the axum surface (handler +
socket↔events adapter); `service.rs` is the pure logic (select a deployment, then
splice client ↔ upstream). The pool itself lives in
`crates/providers/src/realtime_pool.rs`.

## Connection pooling

### The problem

The gateway's realtime overhead lives **entirely in session establishment**. On each
client connect it dials a *fresh* upstream WS to OpenAI and waits for
`session.created` before it can serve. Measured at 5000 calls / 500 concurrency, the
fresh-dial session phase is **~360 ms** vs **~7 ms** direct; dial, first-audio, and
streaming add ~0. So the one lever is removing that per-connect handshake from the
critical path.

### The idea

Keep a few upstream OpenAI sockets **already connected and already past
`session.created`** (buffered). On a client connect, hand off a warm socket — relay
its buffered `session.created` instantly (a local `Vec::pop`, sub-millisecond) and
splice exactly as a fresh dial would. A background task keeps the pool topped up. On
a miss or dead socket we fall back to fresh-dial: the pool is a latency optimization,
never a correctness dependency.

```
                          ┌───────────────────────────────────────┐
   client connect ──────► │ routes/realtime → service::run         │
                          │   pool.take(key)                        │
                          │     hit  → relay buffered               │
                          │            session.created, then splice │
                          │     miss → fresh dial (original path)    │
                          └───────────────┬───────────────────────┘
                                          │ replenish (async, concurrent)
                          ┌───────────────▼───────────────────────┐
   background task ─────► │ RealtimePool: per-key warm sockets      │
                          │   each = { ws, buffered session.created}│
                          │   liveness-checked before handoff       │
                          └─────────────────────────────────────────┘
```

A warm session is indistinguishable from a fresh one: OpenAI sends `session.created`
unprompted on connect, we pre-read exactly that one frame and relay it on handoff,
and we send nothing else on the socket before a client exists — so the client's first
`session.update` behaves identically either way.

### Sizing

Each warm socket serves **exactly one** session (realtime isn't multiplexed), so the
pool is sized to the **peak concurrent connects per instance**, not total live
connections:

```
REALTIME_POOL_SIZE ≈ peak_concurrency / instance_count
```

e.g. 500 concurrency over 10 instances → ~50–64 per instance. The replenisher dials
the missing sockets **concurrently**, so a drained pool refills in ~one handshake
window and keeps supply close to the connect rate. Over-provisioning just burns idle
upstream sockets, which is why warm sockets are short-lived
(`REALTIME_POOL_MAX_IDLE_SECS`).

### Config

| env                           | default | meaning                                                         |
| ----------------------------- | ------- | --------------------------------------------------------------- |
| `REALTIME_POOL_SIZE`          | `4`     | target warm sockets per key. `0` disables pooling (fresh-dial). |
| `REALTIME_POOL_MAX_IDLE_SECS` | `30`    | max time a warm socket sits before it's closed and replaced.    |

### Notes

- **Miss / dead socket → fresh dial.** Burst beyond warm supply, or a socket that
  died, never blocks or fails — it falls back to the original path. The pool can only
  make a connect faster, never slower or more fragile.
- **Auth scope.** The pool key includes `api_key`, so a warm socket is only handed to
  a request resolving to the same key — no cross-tenant reuse.
- **Idle billing.** Warm sockets are liveness-checked at handoff and capped at
  `REALTIME_POOL_MAX_IDLE_SECS` to bound idle billing and dodge OpenAI's idle timeout.
- **Replenish backoff.** If a key's warm-up dials all fail (invalid credentials, an
  unreachable upstream), the replenisher puts that key into exponential backoff
  (500 ms → 30 s cap) instead of re-dialing it every tick. This bounds connection
  attempts against a broken key so it can't exhaust upstream rate limits and degrade
  valid cold-path traffic; the backoff resets the moment a dial succeeds.

Benchmarks and repro: `../../benchmarks/realtime/README.md`.
