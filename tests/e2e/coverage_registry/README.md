# e2e coverage registry

This directory is the **denominator** for e2e test coverage: the set of behaviors we
want covered, one row per behavior, checked into the repo so coverage is a number we
can track instead of a guess. It implements the plan in the "E2E Coverage Tracking"
note; the naming grammar lives in `tests/e2e/CLAUDE.md`.

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

Live pass/fail is separate: each finished pytest node prints an `E2E_RESULT`
logfmt line (see `tests/e2e/e2e_result_reporter.py` and
`tests/e2e/grafana/status_history_panels.md`). Coverage answers "is there a
test for this cell?"; `E2E_RESULT` answers "did that run pass?"

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
