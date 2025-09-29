# LiteLLM Scenarios

These scripts exercise the fork’s three headline integrations (mini-agent,
codex-agent, parallel `acompletion`). They are **live** checks, so run them
manually once the necessary providers or local services are available.

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
