# ai-gateway/src/python — Python interop (load-time only)

Functions here embed the Python interpreter (pyo3) and take the GIL to call into
`litellm` (e.g. read the proxy `model_list`). Compiled only under the
`python-config` feature.

## Hard rule: non-hot-path functions only

Everything in this folder MUST run **at most once per process lifetime — at
startup / load time** (config read, warm-up). NEVER call into Python on the
request path:

- No GIL acquisition per request, per connection, or per realtime event.
- No Python call inside a route handler, the router's hot path, or any loop that
  scales with traffic.

**Why:** the GIL serializes execution and would cap throughput; the realtime data
path must stay pure Rust. Every acquisition is recorded by `crate::gil` — poll
`GET /health/gil`, and `total_acquisitions` MUST stay flat under load.

## How to add one

Resolve whatever Python-derived data you need **once at boot** and hand the rest
of the gateway an owned, plain-Rust value (e.g. build a `Router` from the
resolved `model_list`). Record the acquisition via `crate::gil::record_acquisition()`
immediately before taking the GIL. If a function would need to run per request,
it does not belong here — move the work to Rust, or pre-resolve it at startup.
