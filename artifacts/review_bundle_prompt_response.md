# Review Bundle Findings — LiteLLM Readiness Smokes

## 1. Regression Snapshot
- Command: `. .venv/bin/activate && READINESS_LIVE=1 STRICT_READY=1 READINESS_EXPECT=all_smokes_core python scripts/mvp_check.py`
- Failure: `tests/smoke_optional/test_agent_proxy_validation_errors.py::test_agent_proxy_headers_precedence`
- Root symptom: headers recorded by the tool invoker are `{}` instead of request overrides (`"X-Env": "B", "X-Req": "C"`).

## 2. Primary Root Cause
- `tests/smoke_optional/test_agent_proxy_headers_and_errors.py` replaces `litellm/experimental_mcp_client/mini_agent/agent_proxy.HttpToolsInvoker` with `_FakeInvoker` and never restores it. Because pytest runs `smoke_optional` tests in a single process, the override survives into the next test and prevents the header precedence contract from being exercised with the real invoker.
- Evidence:
  - `tests/smoke_optional/test_agent_proxy_headers_and_errors.py:43-73` constructs `_FakeInvoker` and assigns it directly to `ap_mod.HttpToolsInvoker`.
  - `tests/smoke_optional/test_agent_proxy_validation_errors.py:15-56` expects `HttpToolsInvoker` to internally pass merged headers to the `httpx.AsyncClient`. With the fake still active, `recorded["headers"]` never updates.

## 3. Proposed Fix (minimal & surgical)
1. Change the earlier test to use `monkeypatch.setattr(ap_mod, "HttpToolsInvoker", _FakeInvoker)` so pytest restores the original class automatically. Alternatively, capture the original class and reassign it in `finally`.
2. Re-run the readiness command above; verify `all_smokes_core` now passes.
3. Consider adding a regression test that asserts the recorded headers contain overrides after the suite finishes, ensuring future overrides remain scoped.

## 4. Non-Deterministic (ND) Lane Status
- `scripts/mvp_check.py:560-573` forces hermetic defaults (echo shim, dummy allowance) unless `ND_REAL=1`.
- Make targets (`Makefile:262-285`) only exercise `all_smokes_core` by default, meaning ND-real coverage never executes unless a developer opts in via `smokes-nd-real`.
- The review plan in `local/docs/01_guides/REVIEW_BUNDLE_PROMPT.md` calls for reinstating ND-real smokes. Until `smokes-nd-real` runs in CI (with Ollama reachable) the deploy gate lacks variance coverage.

## 5. Additional Notes
- `readiness.yml` still contains duplicated `code_loop_python` blocks and legacy `all_smokes` entries; keeping the config lean will make future lane splits clearer.
- The FastAPI shim’s header merge path (`litellm/experimental_mcp_client/mini_agent/agent_proxy.py:170-224`) behaves correctly when the genuine `HttpToolsInvoker` is in place, so no shim change is required once the test isolation is fixed.

## 6. Immediate Next Steps for Maintainers
1. Patch the leaking monkeypatch (see §3) and confirm strict readiness passes.
2. Wire `smokes-nd-real` (with `ND_REAL=1`) into a deploy job so ND variance is restored without destabilizing PRs.
3. Deduplicate `readiness.yml` entries and document the lane split in `PROJECT_READY.md` for future reviewers.

---
Prepared: 28 Sep 2025
