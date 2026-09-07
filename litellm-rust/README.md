# LiteLLM Rust

This workspace contains the staged Rust implementation for LiteLLM.

`litellm-core` is the LiteLLM SDK in Rust: one entrypoint per top-level call
that makes the LLM call and hands back a typed response, the same shape as
`litellm.messages()` in Python.

```rust
let response = litellm_core::messages::messages(MessagesRequest {
    model: "claude-sonnet-4-5",
    body,
    api_key: Some(key),
    ..
})
.await?;
```

Python continues to own configuration, retries, routing policy, logging,
callbacks, spend tracking, and customer plugins until each Rust path has parity
coverage and production evidence.

## Crates

| Crate | Role |
|-------|------|
| litellm-core | The SDK. Per-route entrypoints (`messages::messages()`), types, provider transforms (modules under `providers/`), provider resolution, auth, the provider HTTP call, and the router. |
| litellm-config | Config-loading boundary. Returns resolved deployments and optionally delegates loading to Python. |
| litellm-ai-gateway | The axum server (behind the `server` feature) and WebSocket hosts. Translates HTTP/WS to core entrypoints; no provider handlers. |
| litellm-python-interop | Domain-neutral PyO3 foundation for GIL handling and typed Python/Serde conversion. |
| litellm-python-bridge | PyO3 cdylib exposing LiteLLM Rust APIs to the Python SDK. Owns API registration, domain wiring, and Python exception mapping. |

Dependency direction is acyclic: config depends on core, the gateway depends on config and core, and the Python bridge depends on the domain layers and Python interop.

## Layout

```text
crates/
  core/           The SDK: route modules + provider transforms.
    src/messages/   mod.rs (entrypoint), types, transformation, prepare, handler, client
    src/providers/anthropic/messages/transformation.rs
  config/         Config loading and resolved deployments.
  ai-gateway/     Axum server + WebSocket hosts; calls core entrypoints.
  python-interop/ Domain-neutral PyO3 conversion and GIL primitives.
  python-bridge/  PyO3 API adapter for Python LiteLLM.
```

The folder shape follows the Python provider tree:
`core/src/providers/<provider>/<route>/transformation.rs`. The bridge exposes one
function per top-level route, mirroring the core entrypoints.

## Checks

### Native OCR Boundary

`LITELLM_RUST=1` selects native OCR before Python provider preparation. With Rust
disabled, the existing Python execution and authentication paths are unchanged

The bridge retains the complete call argument dictionary as a Python object,
including opaque callback and metadata objects. It creates callback-visible
request dictionaries with shared parameter references, then reads the execution
roots after pre-call dispatch. Mistral retains the original document; Azure and
Vertex Mistral use a shallow document copy, and Vertex DeepSeek projects it into
chat messages using the existing Rust transform. Rust performs provider preparation,
encoding, HTTP and response normalization. Python continues to dispatch existing
logging operations and construct the public response object

This is an opt-in implementation scaffold, not full OCR parity. Azure Mistral and
Vertex Mistral accept inline data URIs with supplied keys/tokens, native environment
keys or auth headers. Azure also accepts a supplied `azure_ad_token`. Vertex
DeepSeek uses its existing chat request and OCR response transforms. Cloud
credential acquisition fails explicitly only when no native credential is available.
HTTP document URL conversion fails only for configs requiring data URIs. Azure
Document Intelligence selects its own config but fails at the polling capability
check before sending a billable analyze request. Cohere transforms, file inputs,
streaming, native response format and compression remain unsupported.
An enabled but missing native extension also fails;
neither case falls back to Python execution. Transport failures currently use a
generic error rather than the SDK's timeout-specific exception

Run the commands under "Checks" in [CLAUDE.md](CLAUDE.md) before pushing Rust
changes. That list is the single source of truth and matches what GitHub Actions
runs for changes under `litellm-rust/`.

### Python-Integrated Tests

From the repository root, run the ignored Cargo tests that need the repository's
Python dependencies and the pinned Ruff checks over the interop crate's Python
test fixtures:

```bash
make test-rust-python
make lint-rust-python-fixtures
```

`test-rust-python` installs the locked SDK dependencies plus the `proxy` extra
(the integration fixtures import `litellm.proxy.*` guardrails, which need
`fastapi`) with uv, points `PYO3_PYTHON` at the project interpreter, and runs
`cargo test -p litellm-python-interop --tests --locked -- --include-ignored`.
`lint-rust-python-fixtures` runs pinned Ruff lint and formatting checks without
syncing the project environment

Run the native OCR acceptance gate from the repository root:

```bash
make test-rust-ocr
```

This builds the current release wheel, installs locked SDK dependencies, the
`dev` test group, and the `proxy` extra in a temporary Python 3.12 environment,
then installs the wheel without resolving dependencies again. The proxy extra
is needed by the shared pytest fixtures. Python isolated mode and pytest's
importlib mode keep the checkout from shadowing the installed wheel

The gate checks that native `ocr` and `aocr` are importable, then runs
`tests/test_litellm/ocr/test_rust_bridge.py` with
`LITELLM_REQUIRE_NATIVE_OCR=1`, so unavailable native OCR fails instead of
skipping. CI uses `make test-rust-ocr RUST_OCR_WHEEL=/absolute/path/to/current.whl`
to test the release wheel it just built. The stdlib-only
`native_route_wheel_test.py` also exercises sync/async OCR through the retained
argument dictionary, including native request preparation and public 429 error
mapping, alongside the other native routes

The Python-integrated Cargo tests validate retained callback identity, mutation,
invocation context and ownership against Python behavior, including existing
LiteLLM components. Short synthetic pre-call contracts use Rust-owned table-driven
cases with inline Python callbacks; larger component scenarios share Python
fixtures. These generic proofs complement, rather than replace, native OCR
acceptance tests

The callback lifecycle scenarios use
`#[serial(python_interpreter)]` to isolate CPython GC and interpreter-wide
LiteLLM settings under `cargo test`. Compatible tests in the same binary use
`#[parallel(python_interpreter)]`: they may overlap each other, but not an
exclusive scenario. Unannotated tests do not participate in this isolation.
Keep the attribute below `#[rstest]` so generated cases acquire it before
fixture setup and Python attachment. Tasks and threads inside each scenario
still run concurrently. Separate test processes have separate interpreters,
so these attributes need no cross-process lock when using nextest
