# ai-gateway — folder architecture

The Axum server that fronts the Rust gateway. It owns transport + config only;
routing lives in the `router` crate, transforms in `core`/`providers`.

```
src/
  main.rs            # entrypoint: build AppState (Router from env), bind, serve
  state.rs           # AppState — shared Arc<Router> handed to every handler
  routes/            # ONE module per route — this is the unit you add to
    mod.rs           #   app(): assembles the axum Router from the handlers
    health.rs        #   GET /health/{liveness,readiness}
    realtime.rs      #   POST /v1/realtime → router.realtime() → providers
```

## Rules

- **One route = one module under `routes/`.** Adding a route means a new file in
  `routes/` plus one `.route(...)` line in `routes/mod.rs`. Nothing else.
- **Handlers stay thin.** A handler extracts input, calls `state.router.<route>()`,
  and maps the result to a response. No provider logic, no transforms, no routing
  decisions in the handler.
- **No business logic here.** Deployment selection is the `router` crate; request/
  response transforms and the upstream call are `core`/`providers`. The gateway is
  HTTP transport + wiring.
- **State is shared and cheap to clone.** Put long-lived handles (the `Router`)
  behind `Arc` in `state.rs`; never rebuild them per request.
- **Config at the edge.** Read env/config only in `main.rs` when building state.
