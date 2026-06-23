# ai-gateway — folder architecture

The Axum server that fronts the Rust gateway. It owns transport + config only;
routing lives in the `router` crate, transforms in `core`/`providers`.

```
src/
  main.rs            # entrypoint: build AppState (Router + gateway key), bind, serve
  state.rs           # AppState — shared Arc<Router> + gateway auth key
  gil.rs             # GIL-activity tracker (records Python acquisitions)
  routes/            # ONE module per route — this is the unit you add to
    mod.rs           #   app(): assembles the axum Router from the handlers
    health.rs        #   GET /health/{liveness,readiness}
    gil.rs           #   GET /health/gil  (poll: is the GIL hit during traffic?)
    realtime.rs      #   POST /v1/realtime → authorize → router.realtime() → providers
  python/            # Python interop (feature: python-config) — see python/AGENTS.md
    mod.rs
    config.rs        #   load_router_from_config() — embeds Python, LOAD TIME ONLY
```

## Rules

- **One route = one module under `routes/`.** Adding a route means a new file in
  `routes/` plus one `.route(...)` line in `routes/mod.rs`. Nothing else.
- **Handlers stay thin.** A handler authenticates, extracts input, calls
  `state.router.<route>()`, and maps the result to a response — no provider
  logic, transforms, or routing decisions in the handler.
- **No business logic here.** Deployment selection is the `router` crate; request/
  response transforms and the upstream call are `core`/`providers`.
- **State is shared and cheap to clone.** Long-lived handles live behind `Arc` in
  `state.rs`; never rebuild them per request.
- **Config at the edge.** Read env/config only in `main.rs` when building state.

## Auth (interim)

`/v1/realtime` requires `Authorization: Bearer <LITELLM_GATEWAY_KEY>` and **fails
closed** when no key is set. The server binds `127.0.0.1` by default (override
with `HOST`) so it is never an open public proxy. This is a stopgap — full
per-key auth, budgets, and rate limits are delegated to the Python proxy in a
later phase. Health routes are unauthenticated.

## Python interop

Anything that calls into Python lives in `python/` and is **load-time only** — see
`python/AGENTS.md`. The realtime data path must never take the GIL.
