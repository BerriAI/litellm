# CLAUDE.md

Rules for `litellm-rust/crates/providers`.

## Responsibility

`providers` owns provider-specific pure transforms. It mirrors the existing
Python provider modules closely enough that parity review is mechanical.

Provider files should map to the Python provider tree:

```text
providers/src/<provider>/<route>/transformation.rs
```

For example, Mistral OCR lives at
`providers/src/mistral/ocr/transformation.rs`, matching
`litellm/llms/mistral/ocr/transformation.py`.

Allowed:
- Provider request transforms.
- Provider response normalization.
- Supported-parameter filtering.
- Provider-specific validation that does not require I/O or secrets.

Not allowed:
- HTTP clients or provider SDK calls.
- Environment variable reads.
- API key resolution or auth header construction.
- Logging, callbacks, spend tracking, retries, routing, cooldowns, or fallbacks.
- Panics on bad user/provider input.

## Required Tests

Every provider transform must include focused unit tests for:
- Supported params matching the Python provider config.
- Unknown params being dropped or transformed the same way as Python.
- Request body shape matching Python output.
- Response normalization with complete, missing, null, and extra fields.
- Bad input returning typed errors.

For OCR specifically, assume documents can contain personal data. Tests should
prove transforms do not copy document contents into error messages.

## Implementation Rules

- Prefer static supported-parameter lists over allocating strings on every call.
- Keep transforms deterministic and allocation-conscious, but choose clarity over
  premature micro-optimization for tiny parameter lists.
- Use typed errors from `core`; avoid stringly-typed error plumbing.
- Add comments only when they explain Python-parity decisions or provider quirks.
- Put route-level provider dispatch in a route file such as `providers/src/ocr.rs`.
  Do not move provider-specific transform logic into the Python bridge.
