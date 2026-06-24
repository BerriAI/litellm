# CLAUDE.md

This file defines the rules for Rust work in LiteLLM.

## Core Boundary

The `core` and `providers` crates describe work; hosts execute work.

Route-level Rust structure mirrors LiteLLM's Python responsibilities:
- `core/src/<route>/` owns the route contract, shared types, and provider
  template traits. For OCR, this means `core/src/ocr`.
- `providers/src/<provider>/<route>/transformation.rs` owns the
  provider-specific transform. For Mistral OCR, this means
  `providers/src/mistral/ocr/transformation.rs`.
- Future network execution belongs in a host/transport layer such as
  `llm_http_handler`, not inside `core` or `providers`.

Allowed in `core` and `providers`:
- Pure request transforms
- Pure response transforms
- Pure stream chunk normalization
- Shared data types and validation errors
- Deterministic token/cost helper logic

Not allowed in `core` or `providers`:
- Network calls
- Environment variable or secret reads
- Filesystem access
- Database or cache access
- Provider SDK signing or auth flows
- Logging callbacks, spend writes, or custom callbacks
- Global mutable runtime state

Python owns rollout state and fallback while Rust is being introduced. Rust
paths must be off by default until parity tests prove equivalence with Python.

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

## Checks

Run these before pushing Rust changes. The same checks run in GitHub Actions
for changes under `litellm-rust/`.

```bash
cd litellm-rust
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

When a Rust path is exposed through Python, add Python parity tests that compare
the existing Python output with the Rust-backed output.
