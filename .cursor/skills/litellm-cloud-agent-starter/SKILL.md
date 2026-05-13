---
name: litellm-cloud-agent-starter
description: Practical Cloud-agent runbook for setting up, running, testing, and perf-checking LiteLLM.
---

# LiteLLM Cloud Agent Starter

Use this skill when a Cloud agent needs to run, test, debug, or perf-check this repository.

## Repo setup

- Work from the repo root: `/workspace`.
- Use the existing virtualenv via `uv run ...`; `uv` is on `PATH` in Cursor Cloud.
- Install only what the task needs:
  - Core/dev: `make install-dev`
  - Proxy/dev: `uv sync --group proxy-dev --extra proxy`
  - Full local tests and Prisma client: `make install-test-deps`
- Before committing Python changes, run `uv run black .`.

## Proxy backend

Start a minimal local proxy with a fake OpenAI-compatible upstream:

```yaml
# config.yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/fake-model
      api_key: fake-key
      api_base: https://fake-api.example.com

general_settings:
  master_key: sk-1234

litellm_settings:
  drop_params: True
  telemetry: False
```

Run it:

```bash
uv run litellm --config config.yaml --port 4000
```

Wait for readiness before testing:

```bash
curl -s http://localhost:4000/health
```

Smoke test:

```bash
curl -s http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{"model":"fake-openai-endpoint","messages":[{"role":"user","content":"hi"}]}'
```

Testing workflow:

- Targeted proxy tests: `uv run pytest tests/test_litellm/proxy/<area> -x -vv`
- Split proxy suite: `make test-unit-proxy-core`, `make test-unit-proxy-guardrails`, or `make test-unit-proxy-misc`
- Fast lint for backend edits: `cd litellm && uv run ruff check .`

Feature flags and paid features:

- Prefer config/env flags over code edits. Common local toggles include `STORE_MODEL_IN_DB=True`, `LITELLM_LICENSE=<mock-or-real-license>`, and `LITELLM_WORKER_STARTUP_HOOKS=<hook-path>`.
- For enterprise-gated behavior, test the non-premium path without `LITELLM_LICENSE` and mock `premium_user` only in focused unit tests.
- For UI settings stored in the DB, use the settings endpoints or test mocks instead of hard-coding defaults.

## Core library and providers

Testing workflow:

- Target one provider or transform: `uv run pytest tests/test_litellm/llms/<provider> -x -vv`
- Core utilities: `make test-unit-core-utils`
- Router changes: `uv run pytest tests/test_litellm/router_utils tests/test_litellm/router_strategy -x -vv`
- Translation/provider contract tests often live under `tests/llm_translation/`.

When provider credentials are unavailable:

- Use mocked unit tests for request transformation, response transformation, streaming chunks, and exception mapping.
- Keep real-provider tests opt-in and guarded by environment variables.

## UI dashboard

The UI source is in `ui/litellm-dashboard/`.

Run the dev UI:

```bash
cd ui/litellm-dashboard
npm run dev
```

Test UI changes:

```bash
cd ui/litellm-dashboard
npx vitest run
```

If the proxy must serve updated static UI assets:

```bash
cd ui/litellm-dashboard
npm run build
cp -r out/* ../../litellm/proxy/_experimental/out/
```

Login and auth workflow:

- Local proxy admin auth uses the configured master key, commonly `sk-1234`.
- Store temporary UI tokens in `sessionStorage`; never write LiteLLM keys to `localStorage`.
- In component tests, mock feature flags through `useFeatureFlags` and mock network calls rather than depending on a running proxy.

## MCP, skills, and agent features

Testing workflow:

- MCP proxy/server tests: `uv run pytest tests/test_litellm/proxy/_experimental tests/test_litellm/experimental_mcp_client -x -vv`
- Skill translation tests: `uv run pytest tests/llm_translation -k skill -x -vv`
- For MCP OAuth/OpenAPI work, map UI-only `openapi` transport to backend `http` before API calls.

## Basic perf testing

Use the Cloud-agent perf runner for quick RPS and overhead checks:

```bash
uv run python scripts/cloud_agent_perf.py \
  --proxy-url http://localhost:4000/chat/completions \
  --proxy-api-key sk-1234 \
  --model fake-openai-endpoint \
  --requests 100 \
  --concurrency 10
```

Compare proxy overhead against a direct provider-compatible endpoint:

```bash
uv run python scripts/cloud_agent_perf.py \
  --proxy-url http://localhost:4000/chat/completions \
  --proxy-api-key sk-1234 \
  --model fake-openai-endpoint \
  --direct-url "$PROVIDER_URL" \
  --direct-api-key "$PROVIDER_API_KEY" \
  --direct-model "$PROVIDER_MODEL" \
  --requests 200 \
  --concurrency 20 \
  --output-json /tmp/litellm_perf.json
```

Perf checklist:

- Warm the endpoint first; keep `--warmup-requests` enabled unless testing cold start.
- Run proxy and direct tests sequentially to avoid shared resource interference.
- Report `rps`, `successful_rps`, p50/p95/p99 latency, and overhead in milliseconds and percent.
- Use small request counts for PR proof and larger repeated runs for regression investigation.

## Updating this skill

When an agent discovers a reliable new runbook step, test workaround, feature-flag setup, or perf trick:

1. Add the shortest reproducible command or workflow to the relevant codebase area.
2. Note required environment variables and whether credentials can be mocked.
3. Keep examples safe for Cursor Cloud and avoid real secrets.
4. Remove stale commands when the repo workflow changes.
