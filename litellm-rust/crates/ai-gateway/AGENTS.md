# ai-gateway — folder architecture

The Axum server that fronts the Rust gateway. It owns transport + config + auth
only; deployment selection lives in `core::router`, transforms in `core`/`providers`.

```
src/
  main.rs            # entrypoint: build AppState (router + master key), bind, serve
  state.rs           # AppState — shared Arc<Router> + master_key
  gil.rs             # GIL-activity tracker (records Python acquisitions)
  auth/              # authentication — imported by routes, never re-implemented
    mod.rs           #   authorize(): single master key (LITELLM_MASTER_KEY) bearer check
  routes/            # one module per route, all matching the same template
    AGENTS.md        #   ← the route template (read this before adding a route)
    mod.rs           #   app(): merges every module's router()
    health.rs        #   simple route: router() + liveness/readiness
    gil.rs           #   simple route: router() + GET /health/gil
    realtime/        #   non-trivial route (the template):
      mod.rs         #     router() — mounts GET /v1/realtime
      handler.rs     #     thin entry: auth, validate, hand off (no logic)
      service.rs     #     business logic (select deployment, call provider) — no axum
      transport.rs   #     WS <-> typed-event adapter
  python/            # Python interop (feature: python-config) — load-time only
    mod.rs, config.rs, AGENTS.md
```

## Rules

- **Routes follow one template.** Each route module exposes
  `pub fn router() -> Router<AppState>`; `routes/mod.rs` only merges them. Simple
  routes are one file; non-trivial routes are a folder (`handler`/`service`/
  `transport`). See `routes/AGENTS.md`.
- **Auth is centralized.** Call `crate::auth::authorize` from a handler; never
  re-implement the check per route.
- **Handlers are thin.** A handler authenticates, validates, and delegates to its
  `service`. No business logic, no provider calls, no transforms in handlers.
- **State is shared and cheap to clone.** Long-lived handles live behind `Arc` in
  `state.rs`; read env/config only in `main.rs` when building state.

## Auth (interim)

A single **master key** (`LITELLM_MASTER_KEY`): any caller presenting it as
`Authorization: Bearer <key>` may invoke the gateway. Fails closed (500) when
unset; constant-time compare. The server binds `127.0.0.1` by default (`HOST` to
override). Full per-key auth + budgets/rate-limits are delegated to the Python
proxy in a later phase. Health routes are unauthenticated.

## Python interop

Anything that calls into Python lives in `python/` and is **load-time only** — see
`python/AGENTS.md`. The realtime data path never takes the GIL.
