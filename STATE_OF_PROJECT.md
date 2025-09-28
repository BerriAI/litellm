Prompt: Generate/Update This Report Reliably

You are an operator agent asked to produce a fresh “State of Our LiteLLM Fork” report. Follow these exact steps and write your answer to STATE_OF_PROJECT.md. Inject the blunt assessment prompt explicitly and answer it in the report.

1) Validate repo & env (local, no network)
   - Confirm we are on our fork branch and tag is present:
     - `git status -sb`
     - `git describe --tags --always || echo no-tag`
   - Run the fork smokes (deterministic):
     - `PYTHONPATH=$(pwd) pytest -q tests/smoke`
     - Expect all green; Node gateway test auto‑skips if Node is absent.
   - Optional live check (requires model key(s)):
     - `PYTHONPATH=$(pwd) python local/scripts/live_checks.py`

2) Verify CI is wired and green
   - Check GitHub Actions workflow `.github/workflows/fork-smokes.yml` ran on the last push to `fork/stable`.
   - Ensure Python 3.12 job is green.

3) Summarize current capabilities (fork‑only paved road)
   - Router refactor seam (flag‑gated), mini‑agent (in‑code, small loop), extras (cache/images/response/batch), import‑time robustness, smokes & Node gateway demo.

4) Blunt assessment (inject prompt, then answer)
   - Prompt to inject verbatim:
     - “With blunt criticalal expert-level honest assessment, How does our fork compare with with the original litellm”

5) HAPPYPATH alignment check (local/docs/01_guides/HAPPYPATH_GUIDE.md)
   - Ensure no new flags beyond the router_core seam, defaults unchanged, determinism in smokes, and copy‑paste commands only.

6) Write the report with sections: Executive Summary, Capabilities, Validation Status, Comparison vs Upstream (Blunt Assessment), Advantages, Gaps, HAPPYPATH Alignment, Iterative Next Steps (Now/Next/Later), Metrics & SLIs, Risks & Mitigations, References.

7) Keep it operator‑friendly
   - Short bullets, copy‑paste commands, minimal fluff.

---

# State of Our LiteLLM Fork — September 24, 2025 (Refreshed @ 2025‑09‑24)

Executive Summary
- We added an experimental Mini‑Agent (with a FastAPI façade), an env‑gated `codex-agent` provider, a pragmatic HTTP tools adapter, and reorganized smokes into deterministic vs live E2E groups. Upstream OpenAI‑compatible APIs remain intact.
- Focus is operator ergonomics and test determinism: `/agent/run` returns a stable envelope, `/ready` supports health checks, and an OpenAI shim (`/v1/chat/completions`) enables quick wiring to Router and provider facades.
- Optional live E2E flows (exec‑rpc, HTTP tools, codex‑agent) are skip‑friendly by default and gated via envs.

Capabilities (fork‑only paved road)
- Mini‑Agent + HTTP Façade
  - Endpoints: `/agent/run`, `/v1/chat/completions` (OpenAI shim), `/ready`.
  - Backends: `local` (in‑process tools), `http` (external tool host), `echo` (hermetic), with final fallback to Router for one‑shot.
  - Envelope: `{ok, final_answer, stopped_reason, messages, metrics}` with `metrics.escalated` and `metrics.used_model`.
  - Tracing: optional JSONL record via `MINI_AGENT_STORE_TRACES=1` and `MINI_AGENT_STORE_PATH`.
  - Files: `litellm/experimental_mcp_client/mini_agent/agent_proxy.py`, `litellm/experimental_mcp_client/mini_agent/__init__.py`.
- HTTP Tools Invoker
  - Lists tools from `/tools`, invokes via `/invoke`, merges env/request headers (Authorization pass‑through), bounded 429 retry.
  - File: `litellm/experimental_mcp_client/mini_agent/http_tools_invoker.py`.
- Local Tools (agent)
  - `exec_python` and `exec_shell` with timeouts; kill()+wait() on timeout; returns `rc/stdout/stderr`.
  - Parallel tool execution keeps original call order in stitched outputs.
- Env‑Gated Provider: `codex-agent`
  - Enable with `LITELLM_ENABLE_CODEX_AGENT=1`.
  - Config via `CODEX_AGENT_API_BASE` (or `api_base`), optional `CODEX_AGENT_API_KEY` → `Authorization: Bearer ...`.
  - Aliases registered: `codex-agent`, `codex_cli_agent`; sync/async HTTP via `httpx`.
  - File: `litellm/llms/codex_agent.py` (docs at `docs/my-website/docs/providers/codex_agent.md`).
- Response Utilities
  - `extract_content`, `assemble_stream_text`, `augment_json_with_cost` for quick, tested helpers.
- Smokes Reorg
  - Deterministic under `tests/local_testing/`; live and optional E2E under `tests/smoke/` and `tests/ndsmoke_e2e/`.
  - Make targets for E2E flows: `e2e-up`, `e2e-run`, `e2e-down`.

Validation Status
- Deterministic suite: `PYTHONPATH=$(pwd) pytest -q tests/local_testing` → green locally.
- Smokes (skip‑friendly): `PYTHONPATH=$(pwd) pytest -q tests/smoke -q` → greens or skips when envs missing.
- E2E (optional): `make e2e-up && make e2e-run && make e2e-down` (requires Docker/compose and services).
- Note: Broader upstream test matrix still requires provider/env setup and will skip/fail fast if absent (expected).

Comparison vs Upstream (Blunt Assessment)
- Prompt (as requested): “With blunt criticalal expert-level honest assessment, How does our fork compare with with the original litellm”
- Answer:
  - The fork is more operator‑friendly for small agentic workflows: a minimal agent façade, an OpenAI shim, and env‑gated provider make prototyping and demos fast without impacting upstream defaults.
  - Stability is stronger on e2e flows we care about: deterministic tests lock response shapes; live paths are clearly gated and skip‑friendly; subprocess handling is safer (timeouts + kill+wait).
  - We trade breadth for paved‑road depth: we didn’t touch upstream gateway auth/rate‑limit breadth; instead we focused on a reliable agent loop, HTTP tools, and a conservative provider adapter.
  - Performance is neutral; the value comes from seams, determinism, and safer envelopes—not raw throughput.

Advantages (vs upstream)
- Clear, small surfaces for agent demos: `/agent/run`, OpenAI shim, `/ready`.
- Safer subprocess tooling; Authorization header handling in tools/provider paths; bounded 429 retry for tools.
- Deterministic envelopes and tests make upgrades less risky.

Gaps (vs upstream)
- `codex-agent` is HTTP‑only here; no CLI/binary integration in this fork.
- Live E2E depends on local services (mini‑agent, exec‑rpc) and Docker; CI should keep these optional.
- We did not expand proxy gateway features (auth/rate‑limits/admin) beyond upstream.

HAPPYPATH Alignment (local/docs/01_guides/HAPPYPATH_GUIDE.md)
- Minimal surfaces, copy‑paste commands, and skip‑friendly tests.
- New functionality is env‑gated (`LITELLM_ENABLE_CODEX_AGENT`, `MINI_AGENT_*`) and off by default.
- Deterministic shapes validated locally; live paths documented and optional.

Iterative Next Steps (Now/Next/Later)
- Now
  - Add `mini-agent` package extras and simple entrypoint (`litellm-mini-agent serve`).
  - Land small doc updates for `codex-agent` and HTTP tools usage; ensure examples run.
- Next
  - Add `/metrics` (optional) to the mini‑agent app for Prometheus; keep import‑guarded.
  - Expand deterministic tests for budget stop (no escalation) and history pruning invariants.
- Later
  - Optional CLI/exec wiring for `codex-agent` when upstream agrees on scope.
  - Harden E2E harness with per‑service readiness checks and time‑bounded retries.

Metrics & SLIs
- Deterministic smokes pass rate (target 100%).
- Mini‑agent `/ready` availability in E2E runs (target 100%).
- Tool call success rate with bounded retries (target > 99% on healthy deps).
- Subprocess timeouts resolved without zombies (observed 0 leaked processes).

Risks & Mitigations
- Live deps flakiness — keep tests skip‑friendly and document envs; add readiness checks.
- Header handling regressions — tests validate Authorization propagation in HTTP tools and provider calls.
- Subprocess safety — enforce timeouts and kill+wait; keep allowlists tight.

References
- Mini‑Agent proxy: `litellm/experimental_mcp_client/mini_agent/agent_proxy.py`
- Mini‑Agent init: `litellm/experimental_mcp_client/mini_agent/__init__.py`
- HTTP tools: `litellm/experimental_mcp_client/mini_agent/http_tools_invoker.py`
- Codex provider: `litellm/llms/codex_agent.py`
- Docs: `docs/my-website/docs/providers/codex_agent.md`
- Smokes: `tests/local_testing/`, `tests/smoke/`, `tests/ndsmoke_e2e/`


Stop/Start/Continue
- Stop: broadening live E2E without readiness checks.
- Start: package the mini‑agent server as an optional extra with a CLI.
- Continue: keep tests deterministic by default and live paths gated/skip‑friendly.

Technical Debt (own it)
- Event‑loop guards in `mini_agent/__init__.py` are test‑oriented; keep minimal and avoid depending on them in library code.
- HTTP tools 429 retry is basic; consider jitter/backoff tuning and max attempt metrics.
- Exec tools allowlist must remain strict; audit periodically.


Canary Parity Plan
- Not applicable for router core in this iteration. Focus canary on mini‑agent e2e availability and tool success rates.


Wiring Draft
- `codex-agent` provider is disabled by default and wires in only when `LITELLM_ENABLE_CODEX_AGENT=1`.
- Mini‑agent server remains optional; docs provide quickstart. No default changes to upstream behavior.

Quick Links
- `README_FORK.md` for fork feature summary and quickstart.
- `docs/my-website/docs/providers/codex_agent.md` for provider usage.
