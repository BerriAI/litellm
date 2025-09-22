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

# State of Our LiteLLM Fork — September 19, 2025 (Refreshed @ 2025‑09‑19)

Executive Summary
- This fork keeps upstream’s public APIs unchanged while adding a paved, opt‑in path for maintainability (router_core seam), a tiny in‑code mini‑agent, and extras to cut boilerplate. Import‑time robustness and deterministic smokes make the fork safer to build on.
- Performance is neutral vs upstream (p50/p95 parity in local harness). The value is stability, ergonomics, and seams for future work—not raw speed.
- CI enforces tests/smoke on `fork/stable`. A canary flag enables the extracted streaming path; legacy remains default.

Capabilities (fork‑only paved road)
- Router Core (opt‑in refactor seam)
  - `Router(..., router_core="legacy"|"extracted")` or `LITELLM_ROUTER_CORE=extracted`.
  - Extracted streaming iterator now exercised fully when flag is set; legacy remains default.
- Mini‑Agent (in‑code helper, not a framework)
  - Local tools: `exec_shell` (allowlist, timeouts), `exec_python` (rc/stdout/stderr, observation injection on failure), pair‑preserving pruning, “research on unsure” nudge.
  - HTTP tools adapter + tiny FastAPI proxy for internal calls.
- Extras (optional, import‑guarded)
  - `extras/cache.configure_cache_redis(router, ...)` — one‑liner Redis wiring.
  - `extras/images.compress_image(...)`, `extras.images.fetch_remote_image(...)` — quick data URLs.
  - `extras/response_utils` — `extract_content`, `assemble_stream_text`, `augment_json_with_cost`.
  - `extras/batch.acompletion_as_completed(...)` — simple concurrent batch helper.
- Robustness
  - fastuuid→uuid fallback; MCP imports guarded to fail lazily on use.
  - Py3.12 loop safety (fixture + small default loop bootstrap in Router).
- Dev hygiene
  - Smokes are deterministic; Node gateway smoke readiness‑polls and skips if Node absent.
  - Internal doc: `docs/dev/router_core_flag.md`. Perf harness: `local/scripts/router_core_perf.py`.

Validation Status
- Local smokes: `PYTHONPATH=$(pwd) pytest -q tests/smoke` → green.
- CI: `.github/workflows/fork-smokes.yml` runs smokes on push/PR into `fork/stable` (Python 3.12). Green on last run.
- Optional live check with model key(s): `PYTHONPATH=$(pwd) python local/scripts/live_checks.py` → returns non‑empty results.

Comparison vs Upstream (Blunt Assessment)
- Prompt (as requested): “With blunt criticalal expert-level honest assessment, How does our fork compare with with the original litellm”
- Answer:
  - Better for developer stability and operator ergonomics: import‑time robustness, deterministic smokes/CI, extras that erase boilerplate, and a tiny mini‑agent for simple tool loops.
  - Maintainability improved a notch: the router seam introduces real “places to put code” without changing defaults. Still conservative; we didn’t explode the monolith.
  - Not faster: runtime performance is essentially neutral. The fork prioritizes safety and seams over micro‑optimizations.
  - Upstreamability: three safe PRs prepared (fastuuid fallbacks, MCP import guards, httpx error tails). The seam/mini‑agent/extras should remain fork‑only until invited.

Advantages (vs upstream)
- Safer imports (no optional dep crashes), test‑friendly event loop behavior, paved extras, and a minimal agent loop for simple iterative flows.
- Clear, opt‑in refactor seam enables future modularization without touching default users.

Gaps (vs upstream)
- No measured speedup; parity is the target. The seam’s main benefits are maintainability and DX.
- The seam is off by default; projects must flip the flag to exercise it.
- Coverage is targeted to our paved‑road features; the full provider zoo/proxy paths remain at upstream breadth.

HAPPYPATH Alignment (local/docs/01_guides/HAPPYPATH_GUIDE.md)
- Minimal surface: one flag (existing seam); no option sprawl.
- Defaults work; overrides are rare; smokes are deterministic; commands copy‑paste cleanly.
- No new backends or global toggles; extras are import‑guarded and optional.

Iterative Next Steps (Now/Next/Later)
- Now
  - Add tiny debug‑only metrics (ttft_ms/total_ms) attached to streaming responses (extracted path) and assert parity ±3% vs legacy in smokes. (Collector tests already present.)
  - Open the three upstream PRs formally (fastuuid fallback, MCP guards, httpx error tails). Keep diffs surgical.
- Next
  - Expand extracted streaming parity tests with mid‑stream fallback and chunk‑ordering assertions. Keep default=legacy.
  - Add 3 short README examples for extras (cache/images/response) + a docs smoke that runs the code blocks.
- Later
  - Consider enabling extracted by default inside our fork only after real‑world parity metrics hold; upstream remains untouched.

Metrics & SLIs
- Smokes pass rate (target 100%).
- Streaming: time‑to‑first‑token and total latency parity legacy vs extracted (±3%).
- Import‑time failures: 0 on CI and local.
- Node gateway readiness failures: 0 (readiness poll).

Risks & Mitigations
- Router drift vs upstream — keep seam opt‑in; keep diffs small; upstream PRs only for safe patches.
- Extras dependency creep — extras remain optional and import‑guarded; no new global knobs.
- Test fragility — continue using local‑only smokes; avoid network; keep readiness polls and autouse loop fixture.

References
- docs/dev/router_core_flag.md — flag usage & safety.
- tests/smoke — deterministic smokes.
- .github/workflows/fork-smokes.yml — CI for our fork.
- local/scripts/router_core_perf.py — perf parity sanity.
- local/scripts/live_checks.py — quick live test harness.


Stop/Start/Continue
- Stop: adding new extras or mini‑agent features until parity and upstream PRs land; keep scope tight.
- Start: capture parity metrics (ttft_ms/total_ms) from a single canary service with LITELLM_ROUTER_CORE=extracted and compare to legacy.
- Continue: running smokes + CI on every push to fork/stable and keeping upstream PRs surgical with “no behavior change for defaults.”

Technical Debt (own it)
- Parallel helpers in router tests: useful for smokes but not a public abstraction. Keep test‑only; do not export.
- Event‑loop bootstrapping in tests: safeguards smokes but should not be relied on by embedders. Keep wrapped in try/except and document as test‑only.
- Metrics attachment via hidden params: handy for debugging; treat as unofficial and avoid downstream coupling. If needed later, expose a formal callback.


Canary Parity Plan
- See local/docs/02_operational/CANARY_PARITY_PLAN.md.
- Gate wiring on: same_text true and worst_ttft/total ≤ 3% over one week.


Wiring Draft
- Branch: feat/extracted-transport-wiring-draft (PR #6 on fork).
- Behavior: wires inner async call to router_core/transport_manager only when LITELLM_ROUTER_CORE=extracted. Default remains legacy.
- Status: Do not merge until canary parity (same_text true; worst ttft/total ≤ 3%) passes for 7 days.

Quick Links
- QUICK_START.md for copy‑paste setup.
- local/docs/02_operational/CANARY_PARITY_PLAN.md for canary runbook.
