# Code Review Request — Mini-Agent Readiness (LiteLLM Fork)

You are an expert-level LLM agent architect and senior developer, specializing in production-readiness, system reliability, and maintainability. Perform a comprehensive code review of the included files. Analyze from three angles simultaneously: (1) immediate runtime risks, (2) long-term architectural flaws, and (3) code quality/maintainability. Find any hallucinated, aspirational, stubbed, or non-working code paths and propose working solutions. Keep fixes minimal (MVP), not over-engineered.

---
## Output Format (one section per file)

---
### File: `[Full Path to File]`

**Overall Assessment:** 1–2 sentence summary of purpose, quality, and main risk.

| 🔴 **CRITICAL / WILL BREAK IN PRODUCTION** |
| :--- |
| **1. [Issue Name]:** Clear description of runtime bug/failure, exact signature (timeouts, blank final answer, wrong tool names, etc.), and the fix. |

| 🟡 **MEDIUM / WILL BITE LATER** |
| :--- |
| **1. [Issue Name]:** Architectural brittleness, perf risks, error-handling gaps, maintainability issues. |

| 🔵 **REFINEMENT / CODE HYGIENE** |
| :--- |
| **1. [Issue Name]:** Style/readability, DRY, small diffs preferred. Provide `git diff` snippets when possible. |

| ✅ **STRENGTHS / GOOD PRACTICES** |
| :--- |
| **1. [Strength Name]:** Solid patterns or safeguards worth keeping. |

---

## Context & Symptoms

- Readiness smoke `tests/ndsmoke/test_mini_agent_compress_runs_ndsmoke.py::test_mini_agent_compress_runs_iterates_live_optional` fails.
  - With `ollama/qwen2.5-coder:14b`: `/agent/run` loops until `httpx.ReadTimeout` (~240s).
  - With `ollama/qwen2.5-coder:3b`: returns 200 but `final_answer==""`; the smoke asserts expected strings are present and fails.
- We patched the loop to:
  - Map ad-hoc tool names to `exec_python`,
  - Surface tool stdout as `final_answer` when possible,
  - Add a local `compress_runs` tool,
  - Persist Ollama weights across rebuilds.
- Despite this, we still see blank answers/timeouts depending on the model.

## What to Review (files included)

1. `litellm/experimental_mcp_client/mini_agent/litellm_mcp_mini_agent.py` (agent loop and local tool invoker)
2. `litellm/experimental_mcp_client/mini_agent/agent_proxy.py` (FastAPI wrapper)
3. `tests/ndsmoke/test_mini_agent_compress_runs_ndsmoke.py` (failing readiness smoke)
4. `local/docker/compose.exec.yml` (docker stack, persistent Ollama volume)
5. `.env` (runtime env used by ndsmokes)

## Repro Notes

- Ensure Ollama has the model(s): `docker exec ollama ollama pull qwen2.5-coder:14b` (and/or `:3b`).
- Persistent cache via compose volume: `${OLLAMA_DATA_DIR:-./local/ollama}:/root/.ollama`.
- Run (examples):
  - `MINI_AGENT_API_HOST=127.0.0.1 MINI_AGENT_API_PORT=8788 DOCKER_MINI_AGENT=1 LITELLM_DEFAULT_CODE_MODEL=ollama/qwen2.5-coder:14b poetry run pytest tests/ndsmoke/test_mini_agent_compress_runs_ndsmoke.py::test_mini_agent_compress_runs_iterates_live_optional -vv -rs`
  - `poetry run pytest tests/ndsmoke -k mini_agent -vv -rs --maxfail=1`

## Review Goals

1. Identify why the agent loop still returns blank `final_answer` or stalls, and propose a minimal, robust fix to make the readiness smoke deterministic.
2. Ensure tool outputs (stdout) are properly surfaced into the assistant transcript and/or final answer.
3. Confirm roles/turns are correct for tool-enabled flows (avoid assistant→assistant sequences).
4. Keep the surface small and maintainable; do not introduce unnecessary complexity.

