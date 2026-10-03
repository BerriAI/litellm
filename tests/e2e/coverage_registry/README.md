# e2e coverage registry

This directory is the **denominator** for e2e test coverage: the set of behaviors we
want covered, one row per behavior, checked into the repo so coverage is a number we
can track instead of a guess. It implements the plan in the "E2E Coverage Tracking"
note; the naming grammar lives in `tests/e2e/AGENTS.md`.

## The model

A **cell** is one customer-noticeable behavior a single e2e test can assert pass/fail
on, for example `llm.chat_completions.bedrock_converse.tool_use.stream.works`. Cells are
grouped `module > feature > test`, with LLM cells split into `Core LLMs` and
`Non-Core LLMs` for dashboarding. Each cell carries a tier (P0/P1/P2), a source, and a
`fail_before_fix` flag.

The rows live in per-prefix YAML files (`llm_*.yaml`, `mgmt.yaml`, `mcp.yaml`,
`reliability.yaml`, `quota_management.yaml`, `logging.yaml`, `guardrail.yaml`,
`other.yaml`) and validate against
the discriminated union in `schema.py`, so an LLM row cannot carry a guardrail field and
vice versa. `llm` rows with `subject_endpoint` of `chat_completions`, `messages`, or
`responses` roll up to `Core LLMs`; all other LLM endpoints roll up to `Non-Core LLMs`.
LLM endpoint, route, and capability values are typed in `schema.py`, so new taxonomy
values require an explicit schema change. `logging` and `guardrail` are two id-prefixes
that roll up into the single `Logging & Guardrails` dashboard module.

A test declares what it covers with a marker:

```python
@pytest.mark.covers("llm.chat_completions.openai.tool_use.stream.works")
def test_openai_streaming_tool_calls(self) -> None:
    ...
```

## The number

`collector.py` diffs the registry against those markers and reports coverage per module.
It is static: a collect-only pass reads the markers, so it runs no test and needs no live
proxy. Whether a covered cell currently passes or fails is a separate, live concern.

A skipped test asserts nothing, so its markers do not count. A cell is covered only when
at least one test pytest would actually run declares it; a cell claimed by both a live
test and a skipped one stays covered. Skip state comes from pytest's own evaluator, so
`skip` and `skipif` resolve exactly as they do in the e2e run, which also means a
`skipif` on an absent credential makes that cell uncovered in the environments where the
test cannot run. Cells left uncovered this way are listed under the headline (and counted
by `litellm_e2e_coverage_skipped_markers`) so an unskipped-pending gap is visible rather
than inflating the number. The one skip the collector cannot see is `pytest.skip()`
called from inside a test body, since it does not exist until the test runs.

```
cd tests/e2e && PYTHONPATH=. python -m coverage_registry.collector
```

Use `--format loki` after the e2e pytest run in the same Kubernetes job/pod to print
structured stdout lines for Loki:

```
cd tests/e2e && PYTHONPATH=. python -m coverage_registry.collector --format loki --strict
```

This emits exactly one `COVERAGE_TOTAL` line and one `COVERAGE_MODULE` line per module
in `MODULE_ORDER`, in that order. Loki uses log-safe `module=` labels from
`LOKI_MODULE_LABELS` (`core_llms`, `management_ui`, etc.) so existing JSON and
Prometheus consumers keep their human-readable module names unchanged.

The headline is overall coverage. The collector also lists markers that point at ids
not in the registry, so a typo or an unenumerated behavior surfaces instead of being
silently dropped.

Use strict mode in CI once existing draft markers are reconciled:

```
cd tests/e2e && PYTHONPATH=. python -m coverage_registry.collector --strict
```

Strict mode exits non-zero on `@pytest.mark.covers(...)` ids that are not checked into
the registry. Add `--fail-on-collection-errors` when the job should also fail on pytest
collection errors.

## Provider x feature matrix: customer-run Bedrock combinations

The provider and feature combinations customers actually run get explicit cells, expanded
here as incidents surface new ones. The current Bedrock set, seeded from a customer's
production shape (regional `us.anthropic.*` inference-profile ids over both chat routes,
provider response headers for AWS-side correlation, and the Test Connection probe for a
responses-mode Bedrock Mantle deployment):

| Cell | Feature | Covering test |
|------|---------|---------------|
| `llm.chat_completions.bedrock_converse.basic.nonstream.works` | regional `us.` id, Converse | `llm_translation/test_chat_completions_regression_e2e.py` |
| `llm.chat_completions.bedrock_converse.basic.stream.works` | regional `us.` id, Converse stream | `llm_translation/test_chat_completions_regression_e2e.py` |
| `llm.chat_completions.bedrock_invoke.basic.nonstream.works` | regional `us.` id, Invoke | `llm_translation/test_bedrock_provider_matrix_e2e.py` |
| `llm.chat_completions.bedrock_invoke.basic.stream.works` | regional `us.` id, Invoke stream | `llm_translation/test_bedrock_provider_matrix_e2e.py` |
| `llm.chat_completions.bedrock_converse.response_headers.nonstream.works` | `llm_provider-*` headers | `llm_translation/test_bedrock_provider_matrix_e2e.py` |
| `llm.chat_completions.bedrock_converse.response_headers.stream.works` | `llm_provider-*` headers, stream | `llm_translation/test_bedrock_provider_matrix_e2e.py` |
| `mgmt.model.test_connection.happy_path` | Test Connection, Bedrock Mantle | `management/test_model_test_connection_e2e.py` |

## Status: this is a draft for review

The cells were enumerated from the codebase and the tiers are a first proposal. Known
things to settle before treating the set as final:

- tiers are proposed, not signed off; 125 P0 is a lot to prove fail-before-fix, so P0 may
  want tightening
- a few cells need a support check or a prune (for example `llm.embeddings.anthropic.*`
  and `reliability.perf.throughput.under_slo`)
- auth is covered in two places (`other.auth.*` and the mgmt authz assertions); the
  boundary needs a decision, and the auth cluster may deserve promotion to its own module
- the P2 "niche" cells each stand in for a large tail of integrations/providers by design,
  so the denominator is deliberately P0-weighted rather than a full inventory

## MCP migration acceptance (LIT-4506)

This matrix tracks the ten security/compatibility guards and the added JWT acceptance
criteria. Registry membership means a desired behavior, not a passing execution
record. Preserve the source/harness commits, SDK version, auth and transport topology,
executed node IDs and live results with each PR. A skipped or unexecuted case remains
pending. Integration contract IDs belong in `tests/integration/contracts.json`, not
`mcp.yaml`; links below connect the two suites without merging their namespaces

| Requirement | Existing regression or owning suite | Remaining acceptance and owner | Required before |
|---|---|---|---|
| Principal discovery | `mcp/test_mcp_key_access_e2e.py`, `mcp/test_mcp_access_group_e2e.py`, `mcp/test_mcp_toolset_enforcement_e2e.py` | Run key/team/org/user controls on every configured replica; the E2E health check intersects with test-owned servers and proves grant visibility only, non-disclosure of unrelated servers is proven by integration contract other.mcp.health.restricted_keys_intersect_grants_in_both_modes (#41731); native/REST parity remains LIT-4506 | Phase 0 and affected authorization changes |
| No self-attached unauthorized grants | Management key authorization tests | Read back unchanged server/toolset/access-group grants after rejected writes, LIT-4502 | Affected grant capability activation |
| UI/API parity | Admin MCP UI suite | Same non-admin actor and permissions across both surfaces, LIT-3644 / LIT-4506 | Affected UI capability activation |
| Server identity routing | Saved-server lifecycle integration and resolver tests | Cold routing, duplicate/unprefixed names, LIT-4500 | Routing changes |
| Same-URL credential isolation | Outbound credential resolver subject/server tests | Observe distinct user/server credentials at actual upstream transport, LIT-4506 with LIT-3467 / LIT-3559 | Affected auth capability activation |
| Fail-closed credentials | LIT-4501 production-path tests; integration lifecycle owns warm credential-removal follow-up | Observe rejection and zero upstream calls after persisted removal; refresh/challenge work remains LIT-4436 / LIT-4422 / LIT-3433 | Phase 0 auth release |
| Upstream session continuity | Existing controlled integration peer is stateless | Observe upstream session ID across operations, LIT-3143 | Stateful upstream activation |
| Real hooks/guardrails | `mcp/test_mcp_guardrail_e2e.py`, LIT-4889 production-path tests | Request-selected direct/virtual pre-call guard integration and removal of void assertions, LIT-4506 | Phase 0 and changed guardrail paths |
| Discovery and execution permissions | Key-denial and principal-toolset E2E; `integration/compatibility/test_persisted_toolsets.py` | Preserve permitted calls and explicit denial, expand remaining protocol combinations, LIT-4506 | Every authorization migration |
| Stateless/stateful combinations | Stateless SDK-peer integration | Explicit incoming/outgoing combinations and supported versions, LIT-3143 / LIT-3559 | Affected transport activation |
| JWT canonical owner and restart | LIT-3794 / LIT-3795 merged regressions | Real process restart, cold cache, header precedence, expired/invalid JWT and inactive user, LIT-4506 | Applicable Phase 0 auth release |
| No gateway JWT upstream | Existing credential resolver/unit evidence | Observe actual upstream headers across users and same-URL servers, LIT-4506 | Applicable Phase 0 auth release |
| Uninterrupted OAuth and aggregate SSO | `mcp/test_mcp_chat_completion_oauth_e2e.py` | Immediate SDK continuation and aggregate SSO, LIT-3467 with LIT-4506 acceptance; missing browser login is not a pass | Applicable Phase 0 auth release |

The initial package does not close LIT-4506. Recheck all applicable rows before the
execution plan's final acceptance step. Keep protocol conformance, canary and rollback
evidence with their owners; a static registry percentage cannot authorize rollout

The E2E consolidation incorporates PR #34055's inventory and PR #35405's remaining
Datadog input-schema guard. The telemetry argument removal and unskipping already
landed in #38640. LIT-5052 requires the schema guard and all three affected real-Datadog
tests, including seeded completion retrieval, to pass before closure. LIT-5749's runtime
fix already landed in #38488; the principal tests add regression evidence
