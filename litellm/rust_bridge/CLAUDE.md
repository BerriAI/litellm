# CLAUDE.md

Rules for `litellm/rust_bridge`.

## Responsibility

This package is the Python-side bridge to optional Rust transforms. It should
route to Rust when explicitly enabled and safely return the existing Python path
when Rust is disabled, unavailable, or unsupported for a provider.

## Naming And Shape

- Keep this package named `rust_bridge`; do not reintroduce a vague `_rust`
  package.
- Organize by LiteLLM route (`ocr/`, future `rerank/`, etc.).
- Keep route entrypoints such as `litellm/ocr/main.py` small. They should only
  ask this package for a Rust-backed config or callable.
- Keep provider rollout explicit with enums or small provider registries.
- For each route, expose a single Python-to-Rust call that passes one payload to
  the PyO3 module, such as `ocr(payload)`. Do not split provider transform
  operations into multiple PyO3 bridge functions.

## Fallback Rules

- Rust paths are off by default.
- Missing PyO3 modules must fall back unless strict mode is enabled.
- Unknown providers must return the original Python config unchanged.
- Tests must cover disabled, enabled, module-missing, and unknown-provider paths.

## Data Handling

- OCR inputs frequently contain personal data. Do not log documents, base64
  payloads, provider response bodies, or secrets.
- Bridge errors should be bounded and sanitized. Do not surface raw upstream
  OCR bodies through Python exceptions.
- Treat blank configuration values as absent at host/config resolution time.
