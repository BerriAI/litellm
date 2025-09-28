# ND-Smokes (Live / Env-gated)

Purpose: validate live integrations and end-to-end behavior against real services (Ollama, Chutes, Node gateway, Dockerized agent).

- Run: `DOCKER_MINI_AGENT=1 pytest tests/ndsmoke -q`
- Skip by default unless env vars indicate services are reachable.
- Keep each test resilient: skip early when dependencies are missing/unreachable.
- Prefer small assertions that still prove the integration works (e.g., non-empty final_answer, correct envelope keys).
