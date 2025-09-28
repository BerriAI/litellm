# Local Testing (Deterministic)

Purpose: fast, deterministic checks for contracts, adapters, small behaviors, and edge-cases. These tests may use seams/monkeypatches and must not rely on external services.

- Scope: adapters, response utils, header pass-through, import guards, mini-agent loop invariants with stubbed providers.
- Run: `pytest tests/local_testing -q`
- Policy:
  - Keep tests small and surgical; avoid duplicating end-to-end smokes.
  - If a local test starts making real network calls, move it to `tests/smoke` or `tests/ndsmoke`.
  - Prefer realistic shapes and minimal stubbing to reflect production behavior.
