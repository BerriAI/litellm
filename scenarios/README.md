# LiteLLM Scenarios

These scripts exercise the fork’s three headline integrations (mini-agent,
codex-agent, parallel `acompletion`). They are **live** checks, so run them
manually once the necessary providers or local services are available.π

## How to Run

1. Activate the repo virtualenv (`source .venv/bin/activate`).
2. Export the environment flags required by the scenario (see below). Each
   script will print a clear skip message if something is missing.
3. Execute the scenario with `python scenarios/<script>.py`.

A convenience target keeps the deterministic order:

```
make run-scenarios
```

## Scenario Index

### `mini_agent_live.py`

- **Purpose**: Ensures the env-gated `mini-agent` provider can execute local
  Python via the MCP shim and return a textual summary.
- **Env**: `LITELLM_ENABLE_MINI_AGENT=1`; optional `SCENARIO_MINI_TARGET_MODEL`
  to override the base LLM. The scenario only exposes the `exec_python` tool to
  avoid hallucinated tool names.
- **Notes**: Requires `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`,
  `scikit-learn` (already pinned in `requirements.txt`).

### `mini_agent_docker_demo.py`

- **Purpose**: Exercises the same provider but routes tool calls into a running
  Docker container.
- **Env**: `LITELLM_ENABLE_MINI_AGENT=1`,
  `LITELLM_MINI_AGENT_DOCKER_CONTAINER=<container>`.
- **Quick start**: `docker compose -f local/docker/compose.exec.yml up --build -d`
  will launch the sandbox container this repo ships under `local/docker/`.

### `codex_agent_router.py`

- **Purpose**: Validates router wiring for multiple providers and the
  env-gated `codex-agent` adapter.
- **Env**: `LITELLM_ENABLE_CODEX_AGENT=1` plus whichever real providers you wish
  to exercise. The script adds optional entries for Gemini and Chutes if the
  associated keys are present. When `LITELLM_DEFAULT_CODE_MODEL` starts with
  `codex-agent/`, be sure to set `CODEX_AGENT_API_BASE` (and optionally
  `CODEX_AGENT_API_KEY`). Otherwise it falls back to the local Ollama model.

### `parallel_acompletions_demo.py`

- **Purpose**: Demonstrates `Router.parallel_acompletions` fan-out and error
  capture.
- **Default model**: `ollama/gemma3:12b` (multimodal). Override via
  `SCENARIO_PARALLEL_MODEL` or upstream defaults if you have a different vision
  stack.
- **Tip**: Because not every Ollama model supports vision, the script prints a
  reminder when responses do not contain visual detail.

## Structured fallback for mini-agent

- `mini_agent_live.py` emits a deterministic `synthetic_summary` JSON object derived from
  the tool output (keys: `peak_change_pct`, `peak_week`, `recommendation`). If parsing
  fails or the tool output is empty, it prints `{"synthetic_summary": {}}`. This lets
  downstream callers rely on a stable schema even if the base model’s final assistant
  message is empty or highly variable.

## Known-good models for tool-calling (and current tips)

- Observed as working well:
  - `qwen2.5-coder:14b` (Ollama) — reliably produces tool calls
  - `openai/gpt-4o-mini` — strong baseline for deterministic tool behavior
  - Llama 3.1 Instruct (tool-tuned variants) — generally solid tool calling
  - Mistral function-call variants — good compliance with tool schemas
- Less reliable without extra nudging:
  - `qwen3:14b` (Ollama) — tends to return `{}` or malformed tool payloads
- Default tuning applied in the scenarios:
  - `tool_choice="required"` to force tool usage when supported
  - `temperature=0` to reduce hallucinated parentheses/syntax errors
  - Where supported, `response_format={"type":"json_object"}` to bias toward JSON
  - A fixed `seed` for providers that honor it (OpenAI/Azure) to reduce variance

### `mini_agent_http_release.py`

- Hits the mini-agent REST API at `/agent/run` using `MINI_AGENT_URL`.
- Falls back to `LITELLM_DEFAULT_MODEL` if `MINI_AGENT_MODEL` is not provided.

### `router_parallel_release.py`

- Minimal fan-out using real credentials (`RELEASE_PARALLEL_API_KEY`).
- Defaults to `LITELLM_DEFAULT_MODEL` if `RELEASE_PARALLEL_MODEL` is unset.

### `router_batch_release.py`

- Exercises `Router.abatch_completion` with real provider credentials.
- Shares the same environment knobs as the parallel release script.

### `image_compression_release.py`

- Compresses the file pointed to by `RELEASE_IMAGE_PATH` using the helper in
  `litellm.extras.images`.

### `chutes_release.py`

- Treats Chutes as an OpenAI-compatible provider via `CHUTES_API_BASE` and `CHUTES_API_KEY`.
- Optionally override the model with `CHUTES_MODEL`; otherwise uses
  `LITELLM_DEFAULT_MODEL`.

### `code_agent_release.py`

- Fires a deterministic `python_eval` tool call using the model dictated by
  `CODE_AGENT_MODEL` or `LITELLM_DEFAULT_CODE_MODEL`.

## Notes on iteration and timeouts

- If weaker models stop early or fail to complete a tool call:
  - Increase `SCENARIO_MINI_MAX_ITER` and/or `SCENARIO_MINI_MAX_SECONDS`
  - Switch to a stronger base model via `SCENARIO_MINI_TARGET_MODEL`
  - Ensure the tool schema is minimal and unambiguous

## Stress tests

Complementary high-load scenarios live in `stress_test/`:

- `parallel_throughput_benchmark.py`
- `parallel_acompletions_burst.py`
- `codex_agent_rate_limit_backoff.py`
- `mini_agent_concurrency.py`

Run them with `make run-stress-tests`. Each script emits live metrics and
respects the same environment variables described above. No mocking occurs, so
ensure the relevant providers are reachable before executing the stress suite.
