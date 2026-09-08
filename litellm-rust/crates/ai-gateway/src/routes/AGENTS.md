# routes/ — the route template

Every route follows one shape: each module exposes `pub fn router() -> Router<AppState>`; `routes/mod.rs::app` merges them all and applies state once. Adding a route means creating the module and adding one `.merge(<name>::router())` line.

## Default: one file

- A route is one file with `router()` + its handler(s) (handlers stay private); `health.rs` is the example
- Don't split until it hurts

## Split out `service` when there's real logic

- Put testable business logic in a sibling `service` (file, or folder if it grows)
- The route file stays the axum surface (router + handler + any socket/SSE adapter); `service` is plain Rust with no axum types
- `service` picks the deployment and calls the `core` entrypoint (see `messages/service.rs`), never builds a provider request, resolves a key, or performs the provider call
- Split further (`transport`, `repo`, ...) only when one file genuinely gets hard to read

## Invariants

- Auth is an extractor, not a manual call
- Handlers contain no business logic; `service` contains no axum types
- No provider handlers here: transforms, auth headers, and the provider HTTP call live in `core/src/<route>/`
- A route owns its paths in its own `router()`; `mod.rs` only merges
- Cross-cutting concerns go in Tower layers in `mod.rs`, not duplicated in handlers
