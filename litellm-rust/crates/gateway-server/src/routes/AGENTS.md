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
`health.rs` is the example.

## Runtime boundary
When a route has transport-neutral orchestration, put it under
`litellm-gateway-inference::runtime` and test it there. The route file stays the Axum
surface: router, handler, and socket or SSE adapter. Never build a provider
request, resolve a provider key, or perform the provider call in this crate.

## Invariants
- **Auth is an extractor, not a manual call.** A handler requires auth by adding
  `crate::auth::RequireMasterKey` to its arguments; it runs during extraction.
  Never re-implement the check per route.
- **Handlers contain no business logic; `litellm-gateway-inference` contains no Axum types.**
- **No provider handlers in this crate.** Transforms, auth headers, and the
  provider HTTP call live in `core/src/<route>/`.
- A route owns its paths in its own `router()`; `mod.rs` only merges.
- Cross-cutting concerns (logging, CORS, timeouts) → Tower layers in `mod.rs`,
  not duplicated in handlers.
