# Routes

Whether a route may be called

## `error.rs`

- Every error a file here can produce lives in `routes/error.rs`. No other file declares its own error type
- `MissingRouteGroup` is a settings build failure from `RouteGroups::new`, never a request denial, so it stays out of `crate::Error`
- Every message is in `#[error(...)]`, built from the variant's typed fields. Keep each opening sentence stable, since clients match on it
- A file that only answers a question (`passthrough.rs`, `managed.rs`, `table.rs`) returns a plain value and adds nothing here
