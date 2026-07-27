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
It is static in the strong sense: a cell is covered when a test declaring it exists in
the tree, so the number is a property of the source and does not move with the runner.
The markers are read off the source with `ast`, which keeps deselected tests (the
`weekly` load test, gated on `E2E_WEEKLY_ANOMALY`) and modules behind an optional
dependency (`pytest.importorskip("mcp")`) counted on every machine. A collect-only
pytest pass runs alongside it for the two things the source text cannot give: markers
built at import time (`pytest.mark.covers(*fn(...))` inside a `pytest.param`) and the
nodeids that failed to import, whose cells are genuinely unknowable. It runs no test
and needs no live proxy. Whether a covered cell currently passes or fails is a
separate, live concern.

The Playwright suite under `tests/e2e/ui/` is TypeScript and emits no pytest markers,
so it declares the cells it covers in `tests/e2e/ui/coverage.yaml` and the collector
unions those ids in. An id there that is not in the registry surfaces as an orphan
marker exactly like a typo in a pytest marker.

Each row names the `spec` it lives in and the Playwright `test` title that proves it,
and the collector resolves both against the tree on every run. Rename or delete that
test and the row stops counting and is listed as a stale declaration, so a claim
cannot outlive the test behind it. Comments are stripped before titles are read, so
a test commented out and forgotten fails the same way a deleted one does, and the
strip is string-aware so a `//` inside a title is never mistaken for a comment
opener. An interpolated title is matched against its
literal segments, so it is still checked as far as it can be, and the check is per
row rather than per file: a dynamic title elsewhere in the spec never exempts a row
that names a literal one. A title with no literal text, or one assembled from
variables, backs no row at all, so the contributor has to give that test a
matchable title instead of the check being waved through.

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
the registry, and on UI declarations whose spec or test title no longer exists. Both are
the same failure: the registry claims something the tree does not support. Add
`--fail-on-collection-errors` when the job should also fail on pytest collection errors.

## Status: this is a draft for review

The cells were enumerated from the codebase and the tiers are a first proposal. Known
things to settle before treating the set as final:

- tiers are proposed, not signed off; 125 P0 is a lot to prove fail-before-fix, so P0 may
  want tightening
- `llm.embeddings.anthropic.basic.nonstream.works` was removed: Anthropic ships no
  embeddings API, litellm has no anthropic embedding handler (the one on the chat
  handler is a `pass` stub and no anthropic row in
  `model_prices_and_context_window.json` carries `mode: embedding`), so the row could
  never pass. `reliability.perf.throughput.under_slo` still needs a support check
- prune a row only when it is genuinely unreachable, with the evidence written down
  here; the denominator is the metric, and shrinking it for any other reason games it
- auth is covered in two places (`other.auth.*` and the mgmt authz assertions); the
  boundary needs a decision, and the auth cluster may deserve promotion to its own module
- the P2 "niche" cells each stand in for a large tail of integrations/providers by design,
  so the denominator is deliberately P0-weighted rather than a full inventory
