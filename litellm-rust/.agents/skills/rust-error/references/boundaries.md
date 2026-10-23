# Boundaries

An error becomes text for someone else in exactly two places, and both are explicit mappings. Nothing else formats an error for a client

| Boundary | File | Maps |
|---|---|---|
| Gateway HTTP response | `crates/gateway-inference/src/error.rs` | `status()` picks the code, `body()` builds the Anthropic or OpenAI envelope |
| Python bridge | `crates/python-bridge/src/errors.rs` and `crates/python-bridge/src/routes/ocr/errors.rs` | `RustUpstreamError(status, body)` for provider answers, `ValueError` for request faults and `RuntimeError` otherwise |

Both follow the same order

```rust
match error.upstream_response() {
    Some((status, body)) => pass the provider's status and body through,
    None => by_fault(error.is_request(), error.to_string()),
}
```

So a new failure only needs the right predicate answers to land correctly on both boundaries. If you find yourself adding an arm for your variant at a boundary, check whether a predicate is missing first

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
