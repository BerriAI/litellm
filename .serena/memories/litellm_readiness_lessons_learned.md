# Lessons Learned — LiteLLM Readiness (Smokes, Env, Router)

Date: 2025-09-28
Scope: LiteLLM fork (codex-agent, readiness harness)

## Root Causes (Why it felt chaotic)
- Address drift: Router assumed a model group ("codex-agent-1"). When missing, it was treated like a hostname → DNS errors ("Name or service not known").
- Env/venv drift: Some steps ran `python`/`pytest` by name (system PATH) instead of the venv → "pytest/python: not found" in some paths.
- Live checks mis-wired: Optional live providers (e.g., Gemini) ran during strict/core gates without credentials → noisy 400s/timeouts.
- Low visibility: Buffered output + quiet pytest hid failing anchors and made stalls look like hangs.

## Correct Taxonomy (one meaning per label)
- Deterministic smokes: Hermetic, fast (≤600s), no network; fail on skips.
- ND smokes: Non-deterministic/local infra (shim, docker); single resolved base; clear SKIP if dependency down; shardable.
- Release smokes: Live providers; opt-in only; require keys; minimal curated set.

## Guardrails (keeps tests on the rails)
- Single base: Resolve `MINI_AGENT_API_BASE` once; force `CODEX_AGENT_API_BASE` to match; print both; STRICT errors on mismatch.
- Router group: Ensure `id: codex-agent-1` exists; if not, autogenerate `.artifacts/router_stub.yaml` that maps to the base. In STRICT, fail clearly if user config lacks group.
- Venv-safe execution: Rewrite `pytest`/`python` to `sys.executable -m pytest` / `sys.executable` at harness boundary; launch uvicorn via `python -m uvicorn`.
- Live gating: Wrap all cloud calls with `LIVE_PROVIDERS=1` and presence of secrets; otherwise SKIP loudly with reason.
- Visibility: Stream subprocess output; write JUnit per check; print CAUSE lines (first failing test anchor); write `PROJECT_READY.md` and `local/artifacts/mvp/mvp_report.json`.
- Time budgets & sharding: Deterministic ≤600s; ND shards ≤1200s each; release smokes minimal and parallel.

## Process Changes
- Readiness policy: Dev gate = deterministic only; Deploy gate = STRICT ND + explicit live providers in `READINESS_EXPECT`.
- Change control: Provider or surface changes must update router mapping, readiness checks, and docs in the same PR.
- PR checklist: Endpoint/group touched? Run STRICT ND. New provider or feature? Add/adjust live smoke + docs.
- Flake quarantine: Two flakes in 7 days → quarantine mark + issue; re‑promote with evidence.

## Concrete Fixes Implemented
- Async transport patch in Router smoke (Client + AsyncClient) → no real DNS.
- Env set before litellm import to register `codex-agent` provider consistently.
- Venv-safe execution & uvicorn via active interpreter.
- Base normalization + CONFIG_ERROR messages for drift.
- Router group autoconfig when missing; STRICT validation when provided.
- PROJECT_READY.md + CAUSE anchors + JUnit.

## Operator Runbook (quick)
- Set base once: `export MINI_AGENT_API_BASE=http://127.0.0.1:8788; export CODEX_AGENT_API_BASE=$MINI_AGENT_API_BASE`.
- Keep live off by default: `export LIVE_PROVIDERS=0`.
- Use venv PATH: `export PATH="$PWD/.venv/bin:$PATH"; hash -r`.
- Core: `READINESS_DISABLE_TIMEOUT=1 PYTEST_XDIST_AUTO_NUM_WORKERS=6 make -s project-ready-core-only-with-docker`.
- ND shards: add `-k 'not shard_b'` / `-k 'shard_b'` and `ALL_SMOKES_ND_TIMEOUT=1200`.

## Definition of Green
- `all_smokes_core` ✅ in ≤12 min; deterministic tests hermetic; no DNS/HTTP 400s.
- ND shards ✅ or explicit SKIP with reason (unless required in `READINESS_EXPECT`).
- Live providers only run (and pass) when keys present and `LIVE_PROVIDERS=1`.

## Anti-Regression
- Enforce single-base and router-group checks in harness.
- Gate live calls; explicit env prints; unit test for AsyncClient patch.
- Keep junit + summary artifacts in CI and fail on STRICT policy violations.
