# gateway-auth-core

The Rust side of `litellm/proxy/auth/core`. It mirrors that package's facets, hosts, inputs, outcomes and answers type for type, so a facet can move from `PythonCore` into Rust without changing what the proxy hands in or gets back. Read `litellm/proxy/auth/core/AGENTS.md` first, its rules apply here unchanged

The crate is pure. It depends on no IO, runtime, HTTP, database or Python crate, and a facet reaches outside itself only through its host trait. Each host trait has its own `Error` type, so a pyo3-backed host and a native host can fail in their own terms while the facet code stays the same

Each Python protocol is a trait here. A core hands out its facets through accessors whose types are associated types on `AuthCore`, the same composition as the Python accessors. Host methods that do IO are `-> impl Future + Send` so facets run under Axum and `tokio::spawn` without `#[async_trait]` boxing. Values a host returns as untyped JSON in Python, such as introspection fields, are `serde_json::Value` here

`NativeCore` implements `AuthCore`. Its OAuth2 facet (introspection and proxy-header mode) and custom auth facet are ported. Every other facet is `Unported` and panics if called, the same as `RustCore` in Python. A facet leaves `Unported` only together with the Python change that routes the proxy to it, and both sides then share one test suite of cases

Python reaches the crate through `crates/python-bridge/src/auth.rs`, never through pyo3 types here. A value that crosses to Python derives serde, and the bridge converts it. Each facet call is a pyo3 class with one `answer_*` method per host call, so `_native.pyi` types the answer to every step. The bridge's host parks each host call until Python answers it. When a serde shape changes, change its TypedDict in `_native.pyi` with it
