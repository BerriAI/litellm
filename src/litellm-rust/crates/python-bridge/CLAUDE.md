# CLAUDE.md

Rules for `litellm-rust/crates/python-bridge`.

## Responsibility

`python-bridge` is the PyO3 boundary between Python LiteLLM and Rust transforms.
Keep this crate thin. It adapts Python objects to Rust payloads and returns
Python-compatible dictionaries.

## Bridge Shape

- Prefer one stable method per top-level LiteLLM route, for example
  `ocr(payload)`.
- Do not add one exported PyO3 function per provider helper unless there is a
  measured reason.
- Provider dispatch belongs in Rust route modules such as
  `litellm_providers::ocr`, not in this PyO3 crate.
- Python owns rollout state and fallback. Rust should return errors; Python
  decides whether to raise or fall back.

## Data Handling

- OCR payloads can contain personal data and large base64 images. Do not log
  payloads or provider responses.
- Avoid copying large payloads more than needed. The current JSON round-trip is
  acceptable for the first scaffold, but future performance work should evaluate
  direct PyO3 conversion before expanding Rust coverage to image-heavy paths.
- Do not expose raw Rust errors that include document contents or upstream
  bodies.

## Tests

- `cargo test --workspace` must compile this crate.
- Python tests must cover bridge disabled, bridge enabled, and module-missing
  fallback behavior for every exposed route.
