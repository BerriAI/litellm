# Realtime upstream connection pool — design

Status: **prototype / RFC**. The pool implementation in this PR is real but wants a
benchmark on a redeploy before we trust the numbers. This doc is the spec; read it
alongside `routes/realtime/` and `crates/providers/src/realtime.rs`.

## The problem

The gateway proxies OpenAI's realtime WebSocket. A benchmark (5000 calls, 500
concurrency) showed the gateway adds **~+115 ms median total** over
direct-to-OpenAI, and that overhead lives **entirely in session establishment**:

| phase            | direct | gateway  |
| ---------------- | ------ | -------- |
| session phase    | ~7 ms  | ~358 ms  |
| first audio      | ~0     | ~0       |
| streaming        | ~0     | ~0       |

On each client connect the gateway dials a **fresh** upstream WS to OpenAI and
waits for `session.created` before it can serve. First-audio and streaming add
nothing. So the single lever is: **remove the per-connect upstream handshake from
the critical path.**

## The idea

Keep a small set of upstream OpenAI realtime sockets **already connected and
already past `session.created`**, so a client connect can be served from a warm
socket instead of paying the dial + handshake. A background task keeps the pool
topped up. On a miss we fall back to the existing fresh-dial path — the pool is a
latency optimization, never a correctness dependency.

```
                         ┌──────────────────────────────────────┐
   client connect ─────► │ realtime route → service::run        │
                         │   pool.take(key)?                     │
                         │     hit  → relay buffered             │
                         │            session.created, splice    │
                         │     miss → fresh dial (today's path)  │
                         └───────────────┬──────────────────────┘
                                         │ replenish (async)
                         ┌───────────────▼──────────────────────┐
   background task ────► │ RealtimePool: per-key warm sockets    │
                         │  each = { ws, buffered session.created}│
                         │  liveness-checked before handoff      │
                         └───────────────────────────────────────┘
```

### Why "warm" means "past session.created"

OpenAI sends `session.created` immediately and unprompted on connect. The expensive
part the benchmark caught is the TLS + WS upgrade + that first server frame. If we
pre-establish the socket *and* pre-read `session.created`, a warm handoff can relay
that buffered frame to the client instantly and then splice exactly as today — the
client cannot tell a warm session from a fresh one.

## Pool design

A `RealtimePool` keyed by `(model, api_key, api_base)` — the tuple that fully
determines an upstream connection. Each warm entry is:

```
WarmConnection {
    ws:               the established tokio-tungstenite stream (split tx/rx),
    session_created:  the buffered session.created RealtimeEvent,
}
```

A background **replenisher** task per key keeps `len < target` by dialing new
sockets up to `target`. `take(key)` pops one warm entry (liveness-checked); if the
pool is empty or the popped socket is dead, it returns `None` and the caller
fresh-dials.

Warm sockets are **short-lived**: a max idle lifetime caps how long a warm socket
can sit before it's proactively closed and replaced, to bound idle billing and dodge
OpenAI's own idle timeout.

### Sizing

Each warm socket serves **exactly one** client session — realtime sessions are not
multiplexed. So the pool size is governed by the **connect rate × warm lifetime**,
not by the number of concurrent connections. If we expect `R` connects/sec and a
warm socket lives `L` seconds before expiry, a target of roughly `R × (handshake
latency)` covers the critical-path window; over-provisioning just burns idle
sockets. Start small (`REALTIME_POOL_SIZE=4`) and tune from real traffic.

## Caveats (read these)

1. **One socket, one session.** Sessions aren't multiplexed. The pool is sized to
   the *connect rate*, not total live connections. A pool of 4 does not mean "max 4
   concurrent calls" — it means "up to 4 connects can skip the handshake before the
   replenisher catches up"; everything else fresh-dials.

2. **`session.created` is pre-read and buffered.** We read exactly that one frame at
   warm time and stash it. On handoff we relay it first, then splice. We must **not**
   over-read (a warm socket should not consume any later frames before a client
   exists).

3. **Idle billing / idle timeout.** OpenAI may idle-timeout or bill an idle session.
   Warm sockets are therefore short-lived and replenished; we cap their warm
   lifetime and liveness-check immediately before handoff. A warm socket that has
   gone away is discarded, not handed to a client.

4. **Pool miss / dead socket → fresh dial.** On burst (demand exceeds warm supply)
   or a dead warm socket, we fall back to the current fresh-dial path. We **never**
   block or fail because the pool is empty. The pool can only make a connect faster,
   never slower or more fragile.

5. **`session.update` semantics unchanged.** A pre-warmed session starts at OpenAI
   defaults, exactly like a fresh one — we send nothing on the socket between
   `session.created` and handoff. The client's first `session.update` behaves
   identically whether the session was warm or fresh.

6. **Liveness check is best-effort.** A socket can die *between* the check and the
   first client frame. That path is already handled: the splice loop surfaces the
   upstream close and the session ends as it would for any mid-session upstream drop.
   The check just avoids the obvious case of handing over a socket we already know is
   dead.

7. **Auth scope.** Warm sockets are dialed with a specific `api_key`. The pool key
   includes `api_key`, so a warm socket is only ever handed to a request resolving to
   the same key — no cross-tenant reuse.

## Config knobs

| env                       | default | meaning                                            |
| ------------------------- | ------- | -------------------------------------------------- |
| `REALTIME_POOL_SIZE`      | `4`     | target warm sockets per key. `0` disables pooling (→ current fresh-dial behavior). |
| `REALTIME_POOL_MAX_IDLE_SECS` | `30` | max time a warm socket sits before it's closed and replaced. |

`idle_timeout` (the existing per-session reaper) is unchanged and applies after
handoff exactly as before.

## Raw-passthrough fast-path (secondary)

The bridge currently serde-parses **every** frame to `RealtimeEvent` and
re-serializes it, in both directions. When no transform is active (the OpenAI config
is pure passthrough today), that parse+reserialize is pure CPU overhead on the hot
streaming path — every audio delta pays it.

**Idea:** forward the raw `Message` bytes when no transform is registered, only
falling back to the typed parse/transform/serialize path when a transform exists for
the event. This keeps the typed transform hook intact (it's the slow path; you opt
in by registering a transform) and skips the work when there's nothing to do.

**Status in this PR:** documented, not implemented. The current seam parses at the
**axum edge** (`bridge` in `routes/realtime/mod.rs` already turns the socket into a
typed `RealtimeEvent` stream before `service::run` sees it), so a raw fast-path
isn't a localized change to `providers` — it needs the bridge to carry the raw bytes
alongside (or instead of) the typed event so the provider can choose. That's a
worthwhile but more invasive change to the `In`/`Out` seam, and it touches the part
of the system the pool also touches. To keep this PR reviewable and the pool change
isolated, raw-passthrough is left as a **documented follow-up**. The streaming phase
is already ~0 in the benchmark, so the latency win here is small relative to the
session-establishment win the pool targets; the passthrough win is CPU/throughput,
not median latency. Sequencing it after the pool lands is the safe call.

## Phased plan

- **Phase 1 (this PR):** the pool, behind `REALTIME_POOL_SIZE` (default 4; `0` =
  off). Warm handoff relays buffered `session.created`; miss/dead → fresh dial. Unit
  tests against an in-process fake OpenAI realtime server. Re-benchmark pending a
  redeploy.
- **Phase 2:** raw-passthrough fast-path (carry raw bytes through the seam; bypass
  parse/reserialize when no transform is registered).
- **Phase 3:** adaptive sizing from observed connect rate; metrics (hit rate, warm
  age at handoff, dial-fallback count) to drive tuning.

## Testing

- Unit tests run the pool against an **in-process fake** OpenAI realtime WS server
  (a `tokio-tungstenite` acceptor that sends `session.created` on connect, then on
  `response.create` replies `response.created` + `response.output_audio.delta` +
  `response.done`). This gives a controllable upstream for pool mechanics.
- The existing `#[ignore]` live test (`realtime_invokes_openai_and_responds`) stays
  the real-OpenAI e2e.
- Cases covered: warm handoff relays the buffered `session.created`; pool miss falls
  back to fresh dial; a dead warm socket is discarded rather than handed over.
