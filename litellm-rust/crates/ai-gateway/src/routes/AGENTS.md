# routes/ — the route template

Every route follows the **same shape** so the layout is predictable. The rule:

> **Each route module exposes `pub fn router() -> Router<AppState>`.**
> `routes/mod.rs::app` merges them all and applies state once. Adding a route is:
> create the module, then add one `.merge(<name>::router())` line.

## Two templates

### Simple route → a single file (`health.rs`, `gil.rs`)
```
pub fn router() -> Router<AppState> { Router::new().route(PATH, get(handler)) }
async fn handler(...) -> impl IntoResponse { ... }   // handlers stay private
```
Use this when the route is just a handler or two with no real logic.

### Non-trivial route → a folder (`realtime/`)
```
realtime/
  mod.rs        # pub fn router(): mounts the path(s). nothing else.
  handler.rs    # the axum entry point — THIN: auth, validate input, hand off.
  service.rs    # the business logic (no axum types).
  transport.rs  # adapters between axum and the service (e.g. WS <-> typed events).
```
Layer responsibilities, never mixed:
- **handler** — extract + `auth::authorize` + validate; return clean HTTP errors
  (401/400/404) *before* any upgrade/streaming; then delegate. No logic.
- **service** — what the route actually does. Plain Rust, framework-agnostic, so
  it's testable without axum.
- **transport** — only when the wire format needs adapting (sockets, SSE, etc.).
  Keeps axum types out of `service`.

## Invariants
- Auth is **not** re-implemented per route — call `crate::auth::authorize`.
- Handlers contain no business logic; `service` contains no axum types.
- A route owns its paths in its own `router()`; `mod.rs` only merges.
