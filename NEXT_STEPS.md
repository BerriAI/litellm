Next steps to align Mini‑Agent, Codex‑Agent, and Parallel Accomplations with the Happy Path

Goal
- Wire the experimental features into the paved “init → run → open → replay” flow with strict defaults, env‑gated experiments, and deterministic smokes.

1) Spec contract (extend v1 minimally; no option sprawl)
- Map existing fields to behavior:
  - execution.concurrency → bound fan‑out using [Router.parallel_acompletions()](litellm/router.py:638) backed by [run_parallel_requests()](litellm/router_utils/parallel_acompletion.py:18) (preserve_order True, return_exceptions True by default).
  - execution.codex_exec: true → enable env‑gated Codex Agent provider. Require:
    - LITELLM_ENABLE_CODEX_AGENT=1 (see [docs/my-website/docs/providers/codex_agent.md](docs/my-website/docs/providers/codex_agent.md:9))
    - optional CODEX_BINARY_PATH or PATH discovery (see [Verify binary](docs/my-website/docs/providers/codex_agent.md:44))
  - execution.autostart_backend/autostart_dashboard → unchanged; orchestrator behavior remains as in guide.
  - observability.backend: arango → unchanged; annotate Router calls with model_group/run_id metadata for traces/scoreboard.
- Optional (strictly minimal):
  - agent.tool_backend: "local" | "http" (defaults local) to choose Mini‑Agent backend described in [docs/my-website/docs/experimental/mini-agent.md](docs/my-website/docs/experimental/mini-agent.md:5).
  - agent.shell_allow_prefixes: ["echo"] default allowlist for LocalMCPInvoker.

2) Orchestrator: implement run decision tree (no new verbs)
- Preflight
  - Validate Arango env as in Happy Path.
  - If execution.codex_exec is true:
    - Check LITELLM_ENABLE_CODEX_AGENT=1; fail fast with a clear message if missing.
    - Check CODEX_BINARY_PATH or which("codex"); fail fast if not found (guide already instructs).
  - If agent.tool_backend == "http": verify httpx import; fail fast with a short hint.
- Build Router and agent
  - Router models:
    - If codex_exec: add a model entry {"model_name": "codex-agent-1", "litellm_params": {"model": "codex-agent/mini"}} (env‑gated per docs).
    - Otherwise: standard model group from defaults (OpenAI/Gemini/etc.).
  - Mini‑Agent runner (default path): call run_mcp_mini_agent from [docs example](docs/my-website/docs/experimental/mini-agent.md:19). Use LocalMCPInvoker(shell_allow_prefixes=…) or HTTP tools per spec.
  - Codex path (codex_exec: true): call router.acompletion against the “codex-agent-1” model per [Usage](docs/my-website/docs/providers/codex_agent.md:28).
- Concurrency fan‑out
  - If execution.concurrency > 1: wrap multiple “approaches” into RouterParallelRequest items and run via [Router.parallel_acompletions()](litellm/router.py:638) which uses [run_parallel_requests()](litellm/router_utils/parallel_acompletion.py:18).
  - Defaults: preserve order True, return_exceptions True. Summarize failures in the scoreboard, don’t crash the whole run.
- Outputs
  - Save spec snapshot under workspace/runs/<run_id>/manifests/spec.yaml (unchanged).
  - Attach run_id into Router request metadata. Expose scoreboard URL + dashboard URL same as guide.

3) Deterministic smokes (wire under local/tests/smoke; fast/offline first)
- Spec → Run smoke
  - A minimal gamified.yaml with execution.concurrency: 2, codex_exec: false, agent.tool_backend: "local".
  - Assert: run prints URLs, saves snapshot, and persists run documents; “final_answer” exists from Mini‑Agent.
- Mini‑Agent HTTP tools smoke (optional dep)
  - Skip if httpx missing. With tool_http_base_url pointing to a tiny local test handler, assert agent returns an answer and stays within max_iterations.
- Codex‑Agent smoke (env‑guarded)
  - Skip if LITELLM_ENABLE_CODEX_AGENT!=1 or no CODEX binary. Short acompletion run (plan → echo → stop) per [docs usage](docs/my-website/docs/providers/codex_agent.md:56).
- Parallel fan‑out smoke
  - Build 2–3 RouterParallelRequest to the same model. Run [Router.parallel_acompletions()](litellm/router.py:638) with preserve_order True/False; assert tuple format (index, response|error) and that failure in one does not fail all when return_exceptions=True.

4) CLI integration (no new verbs)
- init: unchanged, but add optional toggles:
  - “Enable Codex execution?” default No → sets execution.codex_exec.
  - “Concurrency?” default 1 → sets execution.concurrency.
  - “Mini‑Agent backend?” default local; if http selected, add tool_http_base_url prompt.
- run: implement the decision tree above; print short preflight summaries (Arango ok, Codex ok, tools ok).
- open/replay: unchanged; ensure run_id filters and replay use the saved spec snapshot.

5) Observability and artifacts
- Ensure Router requests include metadata: model_group, run_id (via Router._update_kwargs_before_fallbacks).
- Save Mini‑Agent transcript (messages + tool calls) under workspace/runs/<run_id>/manifests/transcript.json for replay/debug.
- On fan‑out, add “approach_id/index” tags to outputs to group scoreboard rows.

6) Docs updates (paved road only)
- Update HAPPYPATH_GUIDE to mention:
  - execution.concurrency and execution.codex_exec semantics (one paragraph each).
  - The Mini‑Agent default backend = local; http backend requires httpx (optional).
  - Preflight messages users should expect (Arango / Codex / Tools).
- Link to:
  - Mini‑Agent usage [docs/my-website/docs/experimental/mini-agent.md](docs/my-website/docs/experimental/mini-agent.md:5)
  - Codex‑Agent enablement [docs/my-website/docs/providers/codex_agent.md](docs/my-website/docs/providers/codex_agent.md:9)

7) Acceptance criteria
- A: “init → run” with default spec (Mini‑Agent local, concurrency=1) finishes with saved snapshot and prints both URLs.
- B: “run” with execution.concurrency=3 executes three approaches via [Router.parallel_acompletions()](litellm/router.py:638), returns three results, and artifacts list three approach outputs.
- C: “run” with execution.codex_exec=true succeeds when flags/binary present; prints a preflight error otherwise without crashing the CLI.
- D: Smokes are deterministic and green on CI (skip appropriately when optional deps/flags absent).

8) Guardrails and defaults (no regressions)
- Codex‑Agent and Parallel Accomplations remain opt‑in; no default behavior changes in core paths.
- Mini‑Agent used only by the orchestrator for this flow; not active elsewhere.
- Keep surface area minimal: one or two spec keys power everything; overrides are rare.

Key implementation anchors
- Mini‑Agent entrypoints: [docs/my-website/docs/experimental/mini-agent.md](docs/my-website/docs/experimental/mini-agent.md:19)
- Codex‑Agent provider (env‑gated): [docs/my-website/docs/providers/codex_agent.md](docs/my-website/docs/providers/codex_agent.md:1)
- Parallel helper + Router wrapper:
  - [run_parallel_requests()](litellm/router_utils/parallel_acompletion.py:18)
  - [Router.parallel_acompletions()](litellm/router.py:638)

This plan keeps the Happy Path verbs unchanged, introduces only two small spec switches (codex_exec, concurrency), and relies on preflight checks and deterministic smokes to guarantee first‑run success. Once these steps are implemented and the smokes are green on CI, the fork is ready for broader adoption with paved‑road defaults and guarded experiments.