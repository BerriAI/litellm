# ND‑Smoke E2E

Purpose: true end‑to‑end examples that exercise live services with small, outcome‑focused assertions. These tests are opt‑in and should always skip cleanly when dependencies are missing.

Conventions
- Markers: `@pytest.mark.ndsmoke` and `@pytest.mark.e2e`
- Time budgets: each test should complete in < 60–90s
- Skips: if envs/ports are missing or unreachable, bail out early with a clear reason

Envs
- Mini‑Agent API: `MINI_AGENT_API_HOST` (default `127.0.0.1`), `MINI_AGENT_API_PORT` (default `8788`)
- Codex‑Agent: `LITELLM_ENABLE_CODEX_AGENT=1`, `CODEX_AGENT_API_BASE`, optional `CODEX_AGENT_API_KEY`
- Exec RPC: `EXEC_RPC_PORT` (default `8790`)

Run
```bash
# bring up what you need first (see Makefile targets)
pytest -q tests/ndsmoke_e2e -m ndsmoke
```
