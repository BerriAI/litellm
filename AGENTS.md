# PROJECT KNOWLEDGE BASE

## OVERVIEW
Embedded LiteLLM fork for Nolgia provider/model support. This `AGENTS.md` is the agent guidance entrypoint for work in this repo.

## NOLGIA INTEGRATION MAP
- `nolgia-api` calls this LiteLLM fork for provider/model orchestration and owns API routing, billing, credits, jobs, and OpenAPI contracts.
- `nolgia.com` exposes supported models and generation states through the API; do not update UI assumptions without matching API behavior.
- `nolgia-cli` consumes the same API contract as web; provider/model changes should surface through generated API clients, not CLI-only logic.
- `infra` may need new secrets, env vars, networking, or Cloud Run runtime settings before a provider/model change works in staging or production.

## FEATURE CHECKLIST
- Use Python/uv-first LiteLLM workflows and keep provider changes covered by meaningful tests.
- Keep comments rare and only when they explain non-obvious behavior.
- For provider/model changes, update LiteLLM tests, then coordinate `nolgia-api` routing/cost/status behavior and OpenAPI exposure.
- If user-facing, update `nolgia.com` display/capability handling and `nolgia-cli` output only after API support exists.
- If runtime config changes, update `infra` stage/prod env and secrets before relying on new provider settings.
