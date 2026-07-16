# CLAUDE.md

This file defines the rules for Rust work in LiteLLM.

## Crates (exactly three — see AGENTS.md)

`litellm-core` describes work; `litellm-ai-gateway` executes it; `litellm-python-bridge`
exposes it to the Python SDK. A crate is a **layer**, not a route — add modules, not crates.

## Core Boundary

`litellm-core` is the pure translation layer; the `litellm-ai-gateway` host executes work.

Route-level Rust structure mirrors LiteLLM's Python responsibilities:
- `core/src/<route>/` owns the route contract, shared types, and provider
  template traits. For OCR, this means `core/src/ocr`.
- `core/src/providers/<provider>/<route>/transformation.rs` owns the
  provider-specific transform. For Mistral OCR, this means
  `core/src/providers/mistral/ocr/transformation.rs`.
- Network execution lives in the host crate `ai-gateway` (`ai-gateway/src/io/`),
  never inside `core`.

Allowed in `core`:
- Pure request transforms
- Pure response transforms
- Pure stream chunk normalization
- Shared data types and validation errors
- Deterministic token/cost helper logic

Not allowed in `core`:
- Network calls
- Environment variable or secret reads
- Filesystem access
- Database or cache access
- Provider SDK signing or auth flows
- Logging callbacks, spend writes, or custom callbacks
- Global mutable runtime state

Python owns rollout state and fallback while Rust is being introduced. Rust
paths must be off by default until parity tests prove equivalence with Python.

## Port parity: mirror Python, do not hand-roll

This workspace ports LiteLLM's Python behavior; it does not reinvent it. When
Python already centralizes a piece of logic, mirror that abstraction and its field names;
do not reimplement a local shortcut that diverges from Python. See the
`porting-python-to-rust` skill and always apply it when adding or changing a Rust
route, provider transform, router, or config resolution.

Provider resolution is the canonical rule: a deployment's provider comes from its
`custom_llm_provider` (in `litellm_params`), resolved through
`litellm_core::get_llm_provider` (the pure port of `litellm.get_llm_provider`).
Never hand-roll `model.split('/')` in a route (no bespoke `split_provider`): that
ignores the explicit `custom_llm_provider` and drifts from Python. `LiteLLMParams`
carries `custom_llm_provider` so it deserializes straight from the proxy
`model_list`. If `get_llm_provider` lacks a Python behavior you need, extend that
one helper to match Python rather than branching in the caller.

## Production Bar

Rust code in this workspace is held to a strict parity and robustness bar from
the first PR:

- Correctness parity is proven with tests. Do not rely on README claims or
  manual inspection for a port that mirrors Python behavior.
- Every provider transform must have unit tests for supported-parameter
  filtering, request body shape, response normalization, missing/null fields,
  and bad-input errors.
- When Rust is exposed through Python, add Python tests that prove disabled,
  enabled, and unavailable-bridge fallback behavior.
- Avoid panics on user/provider input. Return typed errors and let the host map
  them to Python exceptions or HTTP responses.
- OCR handles documents that often contain personal data. Do not log document
  contents, base64 payloads, provider response bodies, or secrets.
- Error messages must be useful but data-minimized. Truncate or sanitize any
  upstream body before it crosses a host boundary.
- Treat empty or whitespace-only credentials, URLs, and config values as absent
  at the host/config resolution layer.
- Preserve Python output shape intentionally. If a field is always serialized as
  `null` for Python parity, leave a short comment explaining that parity choice.

## Host I/O Rules

These rules apply when adding future crates or modules that execute network I/O,
such as `ai-gateway`, router hosts, or standalone servers:

- Set connect and full-request timeouts. No unbounded waits.
- Reuse HTTP clients; do not construct clients per request.
- Prefer rustls TLS for portable Python wheels and Linux images unless there is
  a documented reason not to.
- Add request IDs and structured tracing at the host layer, without logging OCR
  document contents or secrets.
- Do not echo raw upstream response bodies to callers. Sanitize and bound them.
- Avoid `expect`/`unwrap` in server startup and request paths unless the panic is
  impossible by construction and documented.

## Constants

Magic numbers and fixed strings go in a crate-level `constants.rs`, never
hardcoded inline — the Rust mirror of Python's `litellm/constants.py`.

- Each crate that needs them has `src/constants.rs` (declared `mod constants;`);
  import from it (`use crate::constants::...`). Don't scatter `const` values at
  the top of feature modules.
- An env-overridable tunable still lives in `constants.rs` as its `DEFAULT_*`
  value; the env read (with fallback to that default) happens at the host/config
  resolution layer, not in `core`/`providers`.
- Exception: a value that is purely local to one function and has no meaning
  elsewhere may stay inline, but prefer `constants.rs` when in doubt.

## Checks

Run these before pushing Rust changes. The same checks run in GitHub Actions
for changes under `litellm-rust/`.

```bash
cd litellm-rust
cargo fmt --check
# the ai-gateway binary + server code is behind the `server` feature
cargo clippy -p litellm-ai-gateway --all-targets --features server -- -D warnings
cargo clippy -p litellm-core -p litellm-python-bridge --all-targets -- -D warnings
cargo test --workspace
```

When a Rust path is exposed through Python, add Python parity tests that compare
the existing Python output with the Rust-backed output.
