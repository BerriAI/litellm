# routes/ — the route template

Every route follows the **same shape** so the layout is predictable. The rule:

> **Each route module exposes `pub fn router() -> Router<AppState>`.**
> `routes/mod.rs::app` merges them all and applies state once. Adding a route is:
> create the module, then add one `.merge(<name>::router())` line.

## Default: one file
A route is a single file containing `router()` + its handler(s) (handlers stay
private). This is the norm — don't split until it hurts.
```
pub fn router() -> Router<AppState> { Router::new().route(PATH, get(handle)) }
async fn handle(...) -> impl IntoResponse { ... }
```
`health.rs` and `gil.rs` are examples.

## Split out `service` when there's real logic
When a route has business logic worth testing without axum, put it in a sibling
`service` (a file, or a folder if the route grows). The route file stays the
**axum surface** (router + handler + any socket/SSE adapter); `service` is plain
Rust with **no axum types**. `realtime/` is the example:
```
realtime/
  mod.rs       # axum surface: router() + handler + the WS<->events adapter
  service.rs   # pure logic: select deployment + call provider (no axum) — testable
```
Split `service` further (or add `transport`, `repo`, …) only once a single file
genuinely gets hard to read.

## Invariants
- **Auth is an extractor, not a manual call.** A handler requires auth by adding
  `crate::auth::RequireMasterKey` to its arguments; it runs during extraction.
  Never re-implement the check per route.
- **Handlers contain no business logic; `service` contains no axum types.**
- A route owns its paths in its own `router()`; `mod.rs` only merges.
- Cross-cutting concerns (logging, CORS, timeouts) → Tower layers in `mod.rs`,
  not duplicated in handlers.
