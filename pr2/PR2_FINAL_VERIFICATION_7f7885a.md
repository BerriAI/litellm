# PR2 final verification report

Head 7f7885ab86ebe27e1343c870970c220d76776722. Base (PR1 tip) a2e1e62893de4ddcf3a26155d8cda15abb955b88. PR1 merge base eb8a2d224e75593899f6af418c0aa5d3b3ce3060.
Verification only. No tracked file was edited and nothing was committed or pushed. Artifacts were uploaded as unattached Git blobs.

## Diff
`git diff --stat a2e1e62893d 7f7885ab86e`: 9 files, 1430 insertions, 1 deletion. Production files: litellm/integrations/prompt_cache_alerts/{__init__,detector,evidence}.py and litellm/proxy/proxy_server.py. No Rust, UI, schema or route file.

## Tips and what ran where
| Tip | Change from the previous tip | Ran here |
|---|---|---|
| 8e01f88e932 | first final candidate | Bugbot 4581492446 expiry repro (confirmed) |
| 5796f53c4cb | two Lua PEXPIRE calls plus regression | expiry regression, Veria baseline measurement |
| 97a66c4d850 | evidence.py serializes the message envelope once, plus tests | Veria regression, 19 real-Redis cases, REAL write/read, full FIXTURE flow, baseline leg |
| 7f7885ab86e | test_lifecycle.py only, 3 lines in and 3 out; litellm/ byte-identical to 97a66c4d850 | units serial and -n 4, ordered pair, listed remote gates, coverage, FIXTURE core flow |

## Tests
| Check | Tip | Result |
|---|---|---|
| Units, serial | 7f7885ab86e | 131 passed, exit 0 |
| Units, -n 4 with coverage | 7f7885ab86e | 131 passed, exit 0 |
| Units, -n 4 with coverage | 97a66c4d850 | 130 passed, 1 failed (lifespan registration test, "Config file not found: /tmp/c.yaml") |
| save-worker test then registration test, one process | 97a66c4d850 / 7f7885ab86e | 1 failed, 1 passed / 2 passed |
| Real Redis, tests/integration/caching/test_prompt_cache_alert_streaks_real_io.py | 97a66c4d850 | 19 passed |
| Expiry regression (hit and proof) from an untracked copy | 8e01f88e932 / 5796f53c4cb | 2 failed, 17 passed / 19 passed |
| Veria regression from an untracked copy | 5796f53c4cb / 97a66c4d850 | 1 failed, 30 passed (3,239,928 serialized bytes against a 36,070 byte request) / 31 passed |

Commands, run from the worktree with PYTHONPATH=$PWD:$PWD/litellm/proxy-extras:$PWD/litellm/enterprise and DATABASE_URL, DIRECT_URL, LITELLM_LICENSE unset:
```
python -m pytest tests/unit/integrations/prompt_cache_alerts tests/unit/proxy/proxy_server/test_lifecycle.py -q -p no:cacheprovider
python -m pytest <same targets> -q -p no:cacheprovider -n 4 --cov=litellm --cov-branch --cov-report=
REDIS_HOST=127.0.0.1 REDIS_PORT=6392 python -m pytest tests/integration/caching/test_prompt_cache_alert_streaks_real_io.py -q -p no:cacheprovider
python -m pytest tests/unit/proxy/proxy_server/test_lifecycle.py::test_save_worker_config_writes_json_to_environ tests/unit/proxy/proxy_server/test_lifecycle.py::test_the_proxy_lifespan_registers_prompt_cache_alerts_and_releases_them_on_exit -p no:cacheprovider
```

## Gates at 7f7885ab86e against a2e1e62893d, exit 0 each
```
ruff format --check <changed litellm files>
python litellm/scripts/check_mcp_operation_boundary.py
(cd litellm && ruff check .)
ruff check --config .github/linters/ruff-tests.toml tests
python litellm/scripts/ruff_strict_gate.py --base a2e1e62893de4ddcf3a26155d8cda15abb955b88
python litellm/scripts/type_discipline_gate.py --base <same>
python litellm/scripts/test_quality_gate.py --base <same>
python litellm/scripts/type_check_gate.py --base <same>       # "every basedpyright rule is within its ceiling"
python tests/code_coverage_tests/check_unbounded_in_lists.py
```
BasedPyright diagnostics introduced per changed file: 0 in the four production files, 23 in test files (real-Redis test 10, test_detector.py 10, test_evidence.py 2, test_lifecycle.py 1).

OpenAPI docs gate: the canonical workflow runs it (.github/workflows/test-linting.yml, `python litellm/scripts/openapi_docs_gate.py --base "$GATE_BASE_SHA"`), so it applies. It was NOT run locally at 7f7885ab86e, 97a66c4d850 or 5796f53c4cb. The last local pass on this PR was at 0f89cd086b5 (exit 0). The run at 8e01f88e932 was stopped before it returned.

## Coverage
Local: 280/288 changed executable lines (97.2%), 87/142 branches in touched functions. See PR2_CHANGED_COVERAGE_7f7885a.md for the per-file table, the branch denominator and the uncovered lines. Codecov was not read by this session for this tip.

## Live flow
REAL means a request forwarded to Anthropic (claude-sonnet-5-5, non-streaming /chat/completions). FIXTURE means a canned reply from the local relay and is not a provider miss. Two workers and one dedicated Redis on both sides. Exactly two REAL requests were made, both at 97a66c4d850: 4634 tokens written, then 4634 read. No REAL request was made at 7f7885ab86e or on the baseline; the pair carries because litellm/ is byte-identical.
Case tables, commands and completion signals: PR2_REPLAY_7f7885a.md (blob 1b9c33997c7b139e2484335145fce53677d092ef). Raw logs: head_97a66c4.jsonl, head_7f7885a.jsonl, base_a2e1e62.jsonl.

## Gaps
- Not rerun at 7f7885ab86e: the 19 real-Redis cases, the Veria and expiry regressions, and the live cases for worker and Redis restart after dismissal, missing usage, stream, never proven, and the 306 s idle expiry. All are 97a66c4d850 results.
- The baseline leg at a2e1e62893d is one FIXTURE step.
- Driver steps wait 0.5 s between requests and 3 s before reading rows. A case whose only signal is a row check is observational. The 306 s case is one of these.
- At 97a66c4d850 the ordered failure needed the save-worker test in the same process. Exporting WORKER_CONFIG alone did not fail the registration test. That mechanism was not traced.
- Not measured: whether Anthropic rejects an oversized body before the callback runs.
- Python 3.10 startup was not verified locally.
- PR1 UI, schema and screenshot evidence is at a2e1e62893d and was not repeated for PR2.

## Final CI and Codecov checked by the publishing agent

The final CI code-quality job passed, including the applicable OpenAPI docs gate. Codecov is complete at 7f7885ab86e with 280/288 changed executable lines covered (97.2%) and repository coverage 90.6%. One unrelated unchanged custom-guardrail execution-timeout test failed in proxy-feature-endpoints, causing its aggregate unit check to fail. That direct guardrail path does not invoke this detector or its startup registration. It was left untouched under the approved scope rule
