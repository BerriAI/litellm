# Boundaries

An error becomes text for someone else in exactly two places, and both are explicit mappings. Nothing else formats an error for a client

| Boundary | File | Reference |
|---|---|---|
| Gateway HTTP response | `crates/gateway-inference/src/error.rs` | [gateway](gateway.md) |
| Python bridge | `crates/python-bridge/src/errors.rs` and `crates/python-bridge/src/routes/ocr/errors.rs` | [pyo3](pyo3.md) |

Both ask a nested error the same questions in the same order

```rust
if let Some((status, body)) = error.upstream_response() { the provider's status and body pass through }
else if error.is_missing_credential() { 401 at the gateway, RuntimeError at the bridge }
else if error.is_request() { 400 at the gateway, ValueError at the bridge }
else { 500 at the gateway, RuntimeError at the bridge }
```

So a new failure in a nested crate only needs the right predicate answers to land correctly on both. If you find yourself adding an arm for your variant at a boundary, check whether a predicate is missing first

The two boundaries are not identical. The gateway has variants of its own for body parsing, an unknown model and an unsupported path, and the OCR bridge has Python parity arms such as `FileNotFoundError` for a missing file. Those arms match the boundary's own variants or the top level of a public enum, never a nested error's variants. A missing API key is 401 at the gateway, `RuntimeError` at the inference bridge and `ValueError` at the OCR bridge, each pinned by a test to what Python's own path does

## What reaches the client

`to_string()` of our error is what a client sees for anything that is not a provider answer. That is why messages are written for a reader, causes are rendered once and reqwest URLs are stripped on entry

Provider bodies are passed through, so bound them. Build `Error::upstream` with `truncate_error_body(&text)` from `litellm_http::request`, which caps the body at 256 characters

```rust
if !status.is_success() {
    return Err(litellm_http::Error::upstream(status.as_u16(), truncate_error_body(&text)).into());
}
```

Never format `Debug` of an error into a response or a Python exception

## Logging

Propagate with context, and let the terminal handler own the one log or trace event for a failed call. Do not log at every `?`. Diagnostic tracing excludes raw error strings and bodies, see [`rust-tracing`](../../rust-tracing/SKILL.md)
