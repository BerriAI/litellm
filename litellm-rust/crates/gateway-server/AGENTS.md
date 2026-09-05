# gateway-server folder architecture

The Axum server that fronts the reusable Rust gateway runtime. It owns HTTP/WS
transport, startup config, auth extractors, and application state only.
Deployment selection and transport-neutral orchestration live in
`litellm-gateway-inference` or `core::router`, and provider calls live behind core route
entrypoints such as `litellm_core::messages::messages`. No provider handler
lives here.

```
src/
  main.rs            # entrypoint: build AppState (router + master key), bind, serve
  state.rs           # AppState — shared Arc<Router> + master_key
  auth/              # authentication as an axum extractor — added to handler args
    mod.rs           #   RequireMasterKey: FromRequestParts, single master key (LITELLM_MASTER_KEY)
  routes/            # one module per route, all matching the same template
    AGENTS.md        #   ← the route template (read this before adding a route)
    mod.rs           #   app(): merges every module's router()
    health.rs        #   simple route (one file): router() + liveness/readiness
    realtime/        #   router() + handler + WS<->events adapter
```

## Rules

- **Routes follow one template.** Each route module exposes
  `pub fn router() -> Router<AppState>`; `routes/mod.rs` only merges them. Simple
  routes are one file; non-trivial routes are a folder (`handler`/`service`/
  `transport`). See `routes/AGENTS.md`.
- **Auth is an extractor.** Add `crate::auth::RequireMasterKey` to a handler's
  args; it runs during extraction. Never re-implement the check per route.
- **Handlers are thin.** A handler validates and delegates to `litellm-gateway-inference::runtime`. No
  business logic, no provider calls, no transforms in handlers.
- **Runtime services call `core`, they don't reimplement it.** The reusable
  service picks the deployment and calls the `core` route entrypoint. Provider
  resolution, auth headers, URL building, and the HTTP call are `core`'s job.
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

Python-backed loading lives in `litellm-config` and is **load-time only**. The
server's `python-config` feature forwards to that crate. The realtime data path
never takes the GIL.
