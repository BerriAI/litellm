# Quick Start (Fork)

This fork adds an opt-in Mini-Agent, an env-gated `codex-agent` provider, and a standardized readiness/smoke harness. The sections below mirror the runnable assets in `scenarios/` so you can reproduce a green run end-to-end.

## 1) Prereqs
- Python 3.10+
- Optional but recommended: Docker + Docker Compose
- Optional: provider keys (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`)
- Put them in `.env`; every script calls `load_dotenv(find_dotenv())` so values auto-load.

## 2) Install
- Local editable install for iteration: `pip install -e .`

## 3) One-command smoke run

```bash
make run-scenarios
```

This wraps `scenarios/run_all.py`, which:
- Loads `.env`
- Ensures `LITELLM_ENABLE_MINI_AGENT=1` and `LITELLM_ENABLE_CODEX_AGENT=1`
- Runs the release scenarios listed below
- Prints a colour summary and exits non-zero if any scenario fails

Set `SCENARIOS_STOP_ON_FIRST_FAILURE=1` to short-circuit on the first failure.

## 4) Mini-Agent (local shim)

Script: `scenarios/mini_agent_http_release.py`

```bash
python scenarios/mini_agent_http_release.py
```

This spins up the in-process shim if port 8788 is free, asks the mini-agent to run a short Python snippet, and prints the final answer along with `additional_kwargs.mini_agent.metrics`. Expect a JSON summary like:

```json
{
  "ok": true,
  "final_answer": "{\"release_scenario\":{\"ping\":\"Feature update...\"}}",
  "metrics": {"iterations": 1, "used_model": "ollama/glm4:latest"}
}
```

## 5) Mini-Agent over Docker tools

Script: `scenarios/mini_agent_docker_release.py`

Requirements:
- `LITELLM_ENABLE_MINI_AGENT=1`
- `LITELLM_MINI_AGENT_DOCKER_CONTAINER=<container name>` pointing at the tools container (the bundled stack exposes `litellm-mini-agent`)
- Container must expose `/tools` and `/invoke` on the default bridge or a shared network

The script refuses to run unless the container is reachable. The repo ships a compose file that launches the mini-agent shim, codex sidecar, and Ollama with the shared language toolchains:

```bash
docker compose -f local/docker/compose.agents.yml up --build -d

LITELLM_MINI_AGENT_DOCKER_CONTAINER=litellm-mini-agent \
python scenarios/mini_agent_docker_release.py
```

By default both the local and Docker mini-agent invokers allow Python, Rust, Go, and JavaScript. Adjust via `LITELLM_MINI_AGENT_LANGUAGES` if you need to tighten or extend the tool surface.

On success it prints the agent conversation and `parsed_tools` emitted by the container.

## 6) Mini-Agent live loopback

Script: `scenarios/mini_agent_live.py`

This hits the FastAPI shim via `Router.acompletion`, confirming the loopback path and reporting iterations/duration. Run it alone for a quick confidence check:

```bash
python scenarios/mini_agent_live.py
```

## 7) Router parallel + batch

Scripts:
- `scenarios/router_parallel_release.py`
- `scenarios/router_batch_release.py`

Each script builds a `Router` with mixed providers and demonstrates the respective helper (`acompletion_parallel` vs batch). Both print the model response plus raw payload for inspection:

```bash
python scenarios/router_parallel_release.py
python scenarios/router_batch_release.py
```

## 8) Codex-Agent provider

Script: `scenarios/codex_agent_router.py`

Ensure the env shim or remote endpoint is running, then:

```bash
export LITELLM_ENABLE_CODEX_AGENT=1
python scenarios/codex_agent_router.py
```

You should see the shim response (`{"name": ..., "arguments": ...}`) followed by the final aggregated text.

## 9) Codex-Agent via Docker sidecar

Script: `scenarios/codex_agent_docker_release.py`

Requirements:
- `LITELLM_ENABLE_CODEX_AGENT=1`
- `CODEX_AGENT_DOCKER_CONTAINER=<container name>` (defaults to `litellm-codex-agent`)
- Optional `CODEX_AGENT_API_BASE`; defaults to `http://127.0.0.1:8077` when using the bundled Compose stack

```bash
docker compose -f local/docker/compose.agents.yml up --build -d

CODEX_AGENT_DOCKER_CONTAINER=litellm-codex-agent \
python scenarios/codex_agent_docker_release.py
```

The script verifies the container is running, checks `/healthz`, then invokes the codex-agent provider via Router.

## 10) Chutes release flow

Script: `scenarios/chutes_release.py`

Requires `CHUTES_MODEL` and matching credentials (see `.env.example`). The script prints request/response JSON and the `provider_specific_fields` returned by Chutes. When credentials are absent it exits gracefully with a skip message.

## 11) Code Agent tool call

Script: `scenarios/code_agent_release.py`

This uses the mini-agent with tool choice `required`, drives the `python_eval` tool, and shows the resulting tool call + response payload. It’s a good verification that tool routing and parsed metadata remain intact.

## 12) Image helper demo

Script: `scenarios/image_compression_release.py`

Runs the helper in `litellm.extras.images` against a sample asset and prints a data URL preview. Change `IMAGE_PATH` in your `.env` to point at a local image if needed.

## 13) Parallel acompletions demo

Script: `scenarios/parallel_acompletions_demo.py`

Showcases the router’s fan-out helper and verifies ordering/metadata. Useful for quick parity checks when adding providers.

## 14) Ready checks & tests

- Deterministic/local smokes: `make project-ready`
- Strict deploy-ready (live smokes, no skips): `make project-ready-live`
- After touching router/agent code, rerun `make run-scenarios` to confirm all live demos still pass.

## 15) Extras (optional helpers)

All extras live under `litellm/extras/` and are import-safe. Highlights:

```python
from litellm.extras.cache import configure_cache_redis
from litellm.extras.json_utils import clean_json_string
from litellm.extras.log_utils import truncate_large_value

# Router cache in one line
configure_cache_redis(router, host="127.0.0.1", port=6379, ttl=600)

# JSON repair fallback (uses json_repair when installed, graceful degrade otherwise)
clean = clean_json_string("{\"ok\": true,,}")

# Logging helper for large payloads
print(truncate_large_value(data, max_chars=512))
```

All helpers accept environment overrides and respect `LITELLM_DISABLE_CACHE` / `LITELLM_LOG_SENSITIVE_FIELDS` where applicable.

## 16) When something fails

- Inspect the individual scenario script to see the exact `Router` or agent call that failed.
- Ensure `.env` is loaded (every script calls `load_dotenv(find_dotenv())`, but missing keys still surface as explicit errors).
- For Docker scenarios, verify `docker ps` shows the named container and that it exposes `/tools` and `/invoke`.

Once you can run `make run-scenarios` and `make project-ready-live` without skips, you have the same coverage we use for release readiness.

## 17) CodeWorld Bridge (provider demo)

Script: `feature_recipes/codeworld_bridge.py`

Use CodeWorld's Bridge API from LiteLLM. Configure:

```bash
export CODEWORLD_BASE=http://codeworld:8000
export CODEWORLD_TOKEN=...   # if CodeWorld enforces tokens

python feature_recipes/codeworld_bridge.py
```

The script prints a structured JSON payload with `summary` and the full
payload for `additional_kwargs["codeworld"]` (duration_ms, run_id, run_url,
artifacts, metrics, scorecard, winner). See CodeWorld's `docs/guides/BRIDGE_API.md`
for exact schemas. In production, wire this as a LiteLLM custom provider so a
single `router.acompletion(model="codeworld", ...)` call returns both a human
summary and the machine-friendly metrics.
