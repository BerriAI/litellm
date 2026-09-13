---
name: rust-migration
description: Port a LiteLLM Python route (e.g. /v1/messages, embeddings, chat) to the Rust bridge, mirroring the OCR reference implementation. Use when migrating a LiteLLM SDK/proxy route to litellm-rust, adding a provider to an existing Rust route, or wiring a config-driven `rust: true` opt-in. Keeps the port off-by-default with a Python fallback and full parity tests.
---

# LiteLLM Rust migration

Port a Python route to Rust the way OCR was done. Rust replaces only the pure provider transform plus the upstream HTTP call. Python keeps rollout state, fallback, callbacks, interceptors, and the proxy. The port is off by default and must reach parity with the Python path, proven by tests, before it ships.

## The reference: OCR

Read these first, every time. OCR is the source of truth; do not invent a new shape.

Python
- `litellm/rust_bridge/ocr.py` - Protocols, `use_litellm_rust()` toggle, env read, `load_rust_ocr()/load_rust_aocr()`, thin `ocr()/aocr()` wrappers
- `litellm/rust_bridge/loader.py` - loads the compiled `_native` PyO3 module, caches absence -> `None`
- `litellm/rust_bridge/__init__.py` - exports
- `litellm/ocr/main.py` - `_RUST_OCR_PROVIDERS`, `_rust_ocr_supported`, `_run_rust_ocr/_run_rust_aocr`, the enable gate, and the Python fallback
- `tests/test_litellm/ocr/test_rust_bridge.py` - injected `RecordingBridge`, disabled/enabled/unavailable-fallback tests

Rust (`litellm-rust/`, exactly three crates - a crate is a LAYER, routes/providers are MODULES)
- `crates/core` (pure): `ocr/{mod,transformation,types}.rs` (trait + typed structs), `providers/<provider>/ocr/transformation.rs`
- `crates/ai-gateway` (I/O host): `ocr/{mod,prepare,handler,client,types,hooks,common_utils,tests}.rs`, `io/ocr.rs` re-export
- `crates/python-bridge` (PyO3 cdylib `_native`): `lib.rs` pyfunctions that JSON-marshal and call `run_ocr(OcrRequest{...})`

Also read the Rust rules before touching code: `litellm-rust/CLAUDE.md`, `litellm-rust/AGENTS.md`, `crates/core/CLAUDE.md`, `crates/python-bridge/CLAUDE.md`, `crates/ai-gateway/AGENTS.md`, and the repo root `CLAUDE.md`.

## Boundary rules (do not cross)

Allowed in Rust `core`: pure request/response transforms, stream chunk normalization, shared typed structs, typed errors, deterministic token/cost helpers.

Not allowed in `core`: network, env var / secret reads, filesystem, db/cache, provider auth signing, logging callbacks, spend writes, global mutable state, panics on user/provider input. Network I/O lives only in `ai-gateway`. The `python-bridge` crate is a thin marshaller with no business logic.

Stays in Python (never port): rollout state, the enable gate, fallback to the Python handler, pre-request hooks, interceptors, short-circuits, cache-control injection, logging setup, `get_llm_provider`, supported-param filtering, `mock_response`, and the FastAPI proxy route. The proxy inherits Rust routing automatically because it funnels into the same SDK function.

Typed contracts: no bare `serde_json::Value` / `String` as a transform input or output across trait boundaries. Parse to typed structs at the host edge.

## Steps

1. Map the Python route end to end. Find the SDK entrypoint, the provider-config dispatch (`ProviderConfigManager.get_provider_*_config`), the provider config class, the base config contract, and the final `base_llm_http_handler.*` network call. Note streaming, retries, and any hooks that run before dispatch.
2. List differences from OCR up front (streaming, HTTP-error retries, multi-turn state, response shape). Surface them to the user before coding; phase streaming separately if present.
3. Python bridge: add `litellm/rust_bridge/<route>.py` mirroring `ocr.py`. Extend the single `use_litellm_rust()` with `<route>=`/`a<route>=` kwargs; add `LITELLM_USE_RUST_<ROUTE>` env read; export from `__init__.py`.
4. Python gate + fallback: in the route's handler, add `_RUST_<ROUTE>_PROVIDERS`, a `_run_rust_<route>` helper that reproduces the Python `validate_environment`/`get_complete_url`/`transform_request` for logging parity, and the enable gate. On `None` from the bridge, fall back to the existing Python path unchanged.
5. Config-driven opt-in: support per-deployment `rust: true` read from `litellm_params`, with the global toggle/env as fallback. Read it per request.
6. Rust `core`: add `core/src/<route>/{mod,transformation,types}.rs` (trait + typed structs) and `core/src/providers/<provider>/<route>/transformation.rs`. Constants go in `constants.rs`, never inline.
7. Rust `ai-gateway`: add `ai-gateway/src/<route>/{mod,prepare,handler,client,types,hooks,tests}.rs` and `io/<route>.rs`. Reuse the shared `http_client()`; set connect + full-request timeouts; never log payloads/secrets; sanitize and bound upstream error bodies.
8. Rust `python-bridge`: add `<route>`/`a<route>` pyfunctions in `lib.rs` and register them in the `_native` `#[pymodule]`. Keep it a thin JSON round-trip over `run_<route>(...)`.
9. Do NOT add a fourth crate and do NOT expose one pyfunction per provider helper. A new route or provider is a module, not a crate. Adding a crate trips `crates/core/tests/workspace_crate_allowlist.rs` on purpose.

## Verification

Python tests (`tests/test_litellm/.../<route>/test_rust_bridge.py`): inject a `RecordingBridge`; pin bridge disabled, enabled, unavailable-fallback, `rust: true` on/off, and provider header/URL/body parity. Write tests that fail if the feature is mutated, not coverage filler.

Rust checks (from `litellm-rust/`):
```bash
cd litellm-rust
cargo fmt --check
cargo clippy -p litellm-ai-gateway --all-targets --features server -- -D warnings
cargo clippy -p litellm-core -p litellm-python-bridge --all-targets -- -D warnings
cargo test --workspace
```

Parity: same request -> Python output == Rust-backed output, byte-for-byte on the response dict. Preserve always-null fields with a comment.

Repo checks before commit: `make pre-commit` (stage changes first), `make lint`. Python max line length is 120.

Proof of fix (per repo CLAUDE.md): curl a live proxy on `localhost:4000` with a `rust: true` deployment hitting the real provider (real `$$$`, not mocks), non-streaming first. Show the command and the output. Do not use pytest output as proof.

## Rollout

Off by default. Phase 1 non-streaming behind the flag with Python fallback intact. Phase 2 streaming (follow `crates/ai-gateway/src/realtime/`). Phase 3 widen the provider set once parity holds.
