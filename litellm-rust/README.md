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

Run the commands under "Checks" in [CLAUDE.md](CLAUDE.md) before pushing Rust
changes. That list is the single source of truth and matches what GitHub Actions
runs for changes under `litellm-rust/`.

### Python-Integrated Tests

From the repository root, use the same entrypoint as the Rust CI workflow to run
the ignored Cargo tests that need the repository's Python dependencies:

```bash
make test-rust-python
make test-rust-python TEST_FILTER=component_contract
make test-rust-python TEST_FILTER=retained
make lint-rust-python-fixtures
```

The test target covers `litellm-python-interop` (including `component_contract`
and `prepared_call`) and `litellm-python-bridge` (including `ocr_retained`).
`TEST_FILTER` is an optional Rust test-name substring, not a Python fixture or
Cargo test-binary name. An unmatched filter runs zero tests, so check the test
counts. The underlying command is:

```bash
cargo test --manifest-path litellm-rust/Cargo.toml \
  -p litellm-python-interop -p litellm-python-bridge --tests --locked -- --ignored
```

Install uv and the repository's pinned Rust toolchain first. The
`install-rust-python-test-deps` prerequisite runs
`uv sync --inexact --frozen --no-default-groups --no-install-project` on each
invocation. This installs the locked SDK dependencies without building or
installing LiteLLM, pulling in the full dev groups, or pruning existing venv
packages. No wheel is needed for these embedded-Python tests; the existing wheel
lane checks the installed public interface separately

The target gets the project interpreter from `uv run --no-sync python`, sets
`PYO3_PYTHON` to that executable, and queries its `sysconfig` for both Python and
platform-specific site-packages. It prepends the repository root and those paths
to `PYTHONPATH`, preserving any existing entries, so embedded Python imports the
checkout and its dependencies. Use uv's `UV_PYTHON` and `UV_PROJECT_ENVIRONMENT`
settings to select a different interpreter or project venv. The target also sets
`LITELLM_LOCAL_MODEL_COST_MAP=True` to use the checked-in model cost map

Fixture checks are separate from the test target so filtered reruns stay focused.
`make lint-rust-python-fixtures` runs pinned Ruff lint and format checks with
`ruff-tests.toml` over both crates' `tests/` directories, including
`crates/python-bridge/tests/fixtures/ocr_retained.py`, without syncing the project
environment. CI runs both targets; its Python path trigger covers `litellm/**`
so changes to OCR, bridge, logging, streaming, and their shared imports rerun the
integrated tests
