# ai-gateway — folder architecture

The Axum server that fronts the Rust gateway. It owns transport + config + auth
only; deployment selection lives in `core::router`, transforms in `core`/`providers`.

```
src/
  main.rs            # entrypoint: build AppState (router + master key), bind, serve
  state.rs           # AppState — shared Arc<Router> + master_key
  gil.rs             # GIL-activity tracker (records Python acquisitions)
  auth/              # authentication as an axum extractor — added to handler args
    mod.rs           #   RequireMasterKey: FromRequestParts, single master key (LITELLM_MASTER_KEY)
  routes/            # one module per route, all matching the same template
    AGENTS.md        #   ← the route template (read this before adding a route)
    mod.rs           #   app(): merges every module's router()
    health.rs        #   simple route (one file): router() + liveness/readiness
    gil.rs           #   simple route (one file): router() + GET /health/gil
    realtime/        #   route with logic → axum surface + a no-axum service:
      mod.rs         #     router() + handler + WS<->events adapter (the axum surface)
      service.rs     #     business logic (select deployment, call provider) — no axum, testable
  python/            # Python interop (feature: python-config) — load-time only
    mod.rs, config.rs, AGENTS.md
```

## Rules

- **Routes follow one template.** Each route module exposes
  `pub fn router() -> Router<AppState>`; `routes/mod.rs` only merges them. Simple
  routes are one file; non-trivial routes are a folder (`handler`/`service`/
  `transport`). See `routes/AGENTS.md`.
- **Auth is an extractor.** Add `crate::auth::RequireMasterKey` to a handler's
  args; it runs during extraction. Never re-implement the check per route.
- **Handlers are thin.** A handler validates and delegates to its `service`. No
  business logic, no provider calls, no transforms in handlers.
- **State is shared and cheap to clone.** Long-lived handles live behind `Arc` in
  `state.rs`; read env/config only in `main.rs` when building state.

## Auth (interim)

A single **master key** (`LITELLM_MASTER_KEY`), enforced by the
`auth::RequireMasterKey` extractor: any caller presenting it as
`Authorization: Bearer <key>` may invoke the gateway. Fails closed (500) when
unset; constant-time compare. The server binds `127.0.0.1` by default (`HOST` to
override). Full per-key auth + budgets/rate-limits are delegated to the Python
proxy in a later phase. Health routes don't add the extractor (unauthenticated).

## Python interop

Anything that calls into Python lives in `python/` and is **load-time only** — see
`python/AGENTS.md`. The realtime data path never takes the GIL.
