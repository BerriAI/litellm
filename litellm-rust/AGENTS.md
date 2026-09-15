# Rust error conventions

Small crates define their public `Error` enum in `src/error.rs` and reexport it with `mod error; pub use error::Error;` from `lib.rs`. A crate implementing another crate's interface may reuse that interface's error instead of inventing a wrapper

In `core`, each route defines `Error` in `src/<route>/error.rs` and reexports it from the route module. Route entrypoints return their route error. The root `core::Error` is a thin enum wrapping route errors for consumers that handle multiple routes

Create error types at boundaries with different ownership, possible failures, or caller handling. Do not create one per file or provider by default. Keep related request, response, and polling enums together in the route's `error.rs` when narrower function contracts justify them. Reexport public payload types from the owning module; keep implementation-only errors private or `pub(crate)`

Shared subsystem errors live beside their implementation and do not depend on routes or the root error. Provider implementations normally return route errors. Add a provider error only when distinct typed handling requires one. Keep small private helper errors inline rather than creating a directory solely for an error file

Import dependency errors from their owning crate, for example `litellm_auth::Error`. Do not reexport them from `core` as `AuthError`. A local import alias such as `use litellm_auth::Error as AuthError;` is appropriate when multiple error types are in scope

Derive `Debug` and `thiserror::Error`. Add `Clone`, `PartialEq`, or `Eq` only when needed; do not stringify causes to enable those derives

Use `#[from]` for unambiguous, context-free wrapping. It also marks the source. Use `#[error(transparent)]` when intentionally forwarding display and source behavior. Use `#[source]` with explicit context when the wrapper adds meaning. Use manual `From` only for infallible, context-free conversions, defined beside the receiving error when practical. Use `map_err` or a named constructor when conversion needs operation, provider, field, or dispatch context

Conversions flow from narrower errors into broader errors. Do not wrap the root error inside a route error, add reverse conversions, or add every transitive conversion solely to make `?` compile. Shared lifecycle interfaces carry the caller's error type

Preserve typed causes and structured status/dispatch information through Rust layers. Sanitize sensitive diagnostics deliberately. Convert to presentation strings and map Python exception classes or compatibility HTTP statuses at host boundaries. Never infer retry or fallback safety from display text; timeouts do not prove that no request was dispatched

Test caller-visible behavior: typed failure details survive propagation, host exception/status mappings remain correct, and post-dispatch failures cannot authorize replay. Do not test filesystem structure alone
