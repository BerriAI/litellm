# Routes

Whether a route may be called. Its failures follow the same convention as `error/`

## `error.rs`

- Every error a file here can produce lives in `routes/error.rs`. No other file declares its own error type
- `RouteNotExposed` is the denial `exposed.rs` returns for a route the proxy's own allowlist leaves out. It is refused before any credential is read, so it stays out of `crate::Error` and reaches a host through `RequestDenial`
- `MissingRouteGroup` is a settings build failure from `RouteGroups::new`, never a request denial, so it stays out of `crate::Error`
- Every message is in `#[error(...)]`, built from the variant's typed fields. Keep each opening sentence stable, since clients match on it
- A file that only answers a question (`passthrough.rs`, `managed.rs`, `table.rs`) returns a plain value and adds nothing here
