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

The inference gateway and the Python bridge's route mapping both receive a nested route error and decide by the same predicates in the same order, so a new failure lands correctly on both once its predicates answer right

| Predicate | Gateway | Bridge |
|---|---|---|
| `upstream_response()` | the provider's status and body | `RustUpstreamError` |
| `is_missing_credential()` | 401 | no class of its own, a `RuntimeError` because `is_request()` answers false |
| `is_request()` | 400 | `ValueError` |
| none of these | 500 | `RuntimeError` |

If you find yourself adding an arm for your variant at a boundary, check whether a predicate is missing first

The two boundaries are not identical. The gateway has variants of its own for body parsing, an unknown model and an unsupported path, and the OCR bridge has Python parity arms such as `FileNotFoundError` for a missing file. Those arms match the boundary's own variants or the top level of a public enum, never a nested error's variants. A missing API key is 401 at the gateway and `RuntimeError` at both bridges, pinned by tests to what Python's own path does

## Declines

A decline means there is no Rust way to do this, and it is its own type, `litellm_core_utils::decline::Decline { code, detail }`, never a variant of a route error. `code` is stable, dotted by layer (`config.setting`, `route`, `body.field`, `auth.plan`, `router.select`, `llms.streaming`) and safe in a header. `detail` is for the log. Declare each one as a `const`, so a malformed code does not compile

It is legal only before the first effect another implementation would repeat, and it comes from two places. Construction returns every decline a config gives, such as router-core's `ConfigError::Declined`. A route's `plan` returns the first one for a request as `Unplanned::Declined`. `execute` takes the plan and returns `Error`, which has no decline variant, so a late decline does not compile. A host call Rust may not serve gets a plan-time half on the host trait (`Host::plan_select`, `Host::plan_invoke`), and the Rust host's implementation of that half is where its decline is written

It bubbles up unchanged to the edge that dispatched the work, and that edge decides what it means: the Python bridge raises `RustBridgeDeclined(code, detail)` so the SDK runs its own path, and the gateway, with nothing behind it, answers 501 naming the code. See [gateway](gateway.md) and [pyo3](pyo3.md) for each side

Anything after the send is terminal. Every route that transforms the provider's answer collapses its errors with `Error::after_send`, which keeps a provider answer and a response fault as they are and nests anything else as the source of an invalid response

## What reaches the client

A boundary renders anything that is not a provider answer with `litellm_error::Report`, so the reader sees every layer of the chain once. That is why messages are written for a reader, each layer leaves its cause to `source()`, and reqwest URLs are stripped on entry. A boundary that must not show the chain, such as the bridge's TLS settings error that names the setting but not the file, renders fixed text of its own and says so in its test

Provider bodies are passed through intact for gateway and Python error mapping. `litellm_http::Error::upstream(status, body)` retains the complete body, but its `Display`, `Debug` and `Report` diagnostics show only the first 256 characters and mark the cut. Do not log the raw body or `RustUpstreamError` args. OCR's `Provider` error also keeps the whole body for Python's `get_error_class`, bounded by the connection's response size limit

Never format `Debug` of an error into a response or a Python exception. A type whose sources may hold secrets derives `veil::Redact`, so its `Debug` is safe to log

## Logging

Propagate with context, and let the terminal handler own the one log or trace event for a failed call. Do not log at every `?`, and do not log an error you also return. Diagnostic tracing excludes raw error strings and bodies, see [`rust-tracing`](../../rust-tracing/SKILL.md)
