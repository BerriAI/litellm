# Boundaries

A boundary is where an error becomes something another program reads: an HTTP response, a Python exception, a stored or logged message. Each boundary has one mapping function per error type it receives, and nothing outside those functions formats an error for someone else

| Boundary | Mapping | Reference |
|---|---|---|
| Inference HTTP response | `Error::status()` and its renderers in `crates/gateway-inference/src/error.rs` | [gateway](gateway.md) |
| Other gateway surfaces | the `IntoResponse` of each `crates/gateway-*` crate's `Error` | [gateway](gateway.md) |
| Python bridge, inference and OCR routes | `route_failure` in `crates/python-bridge/src/errors.rs` and in `crates/python-bridge/src/routes/ocr/errors.rs` | [pyo3](pyo3.md) |
| Python bridge, everything else | one `From<Error> for PyErr` or mapping function per crate error, all in `crates/python-bridge` | [pyo3](pyo3.md) |
| Python host runtime | `From<Error> for PyErr` in `crates/host-python/src/error.rs` | [pyo3](pyo3.md) |

## Asking the nested error

The inference gateway and the Python bridge's route mapping both receive a nested route error and decide by the same predicates, so a new failure lands correctly on both once its predicates answer right

| Predicate | Gateway | Bridge |
|---|---|---|
| `is_unsupported()` | 501 | `ValueError`, since `is_request()` also answers true |
| `upstream_response()` | the provider's status and body | `RustUpstreamError` |
| `is_missing_credential()` | 401 | no class of its own, a `RuntimeError` because `is_request()` answers false |
| `is_request()` | 400 | `ValueError` |
| none of these | 500 | `RuntimeError` |

If you find yourself adding an arm for your variant at a boundary, check whether a predicate is missing first

The two boundaries are not identical. The gateway has variants of its own for body parsing, an unknown model and an unsupported path, and the OCR bridge has Python parity arms such as `FileNotFoundError` for a missing file. Those arms match the boundary's own variants or the top level of a public enum, never a nested error's variants. A missing API key is 401 at the gateway and `RuntimeError` at both bridges, pinned by tests to what Python's own path does

## What reaches the client

A boundary renders anything that is not a provider answer with `litellm_error::Report`, so the reader sees every layer of the chain once. That is why messages are written for a reader, each layer leaves its cause to `source()`, and reqwest URLs are stripped on entry. A boundary that must not show the chain, such as the bridge's TLS settings error that names the setting but not the file, renders fixed text of its own and says so in its test

Provider bodies are passed through, so they are bounded where they enter our types. `litellm_http::Error::upstream(status, body)` keeps the first 256 characters and marks the cut, so no caller can forget. OCR's `Provider` error keeps the whole body for Python's `get_error_class`, bounded by the connection's response size limit instead

Never format `Debug` of an error into a response or a Python exception. A type whose sources may hold secrets derives `veil::Redact`, so its `Debug` is safe to log

## Logging

Propagate with context, and let the terminal handler own the one log or trace event for a failed call. Do not log at every `?`, and do not log an error you also return. Diagnostic tracing excludes raw error strings and bodies, see [`rust-tracing`](../../rust-tracing/SKILL.md)
