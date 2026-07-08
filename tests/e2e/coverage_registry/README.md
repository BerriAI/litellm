# e2e coverage registry

This directory is the **denominator** for e2e test coverage: the set of behaviors we
want covered, one row per behavior, checked into the repo so coverage is a number we
can track instead of a guess. It implements the plan in the "E2E Coverage Tracking"
note; the naming grammar lives in `tests/e2e/CLAUDE.md`.

## The model

A **cell** is one customer-noticeable behavior a single e2e test can assert pass/fail
on, for example `llm.chat_completions.bedrock_converse.tool_use.stream.works`. Cells are
grouped `module > feature > test`, six dashboard modules in all. Each cell carries a
tier (P0/P1/P2), a source, and a `fail_before_fix` flag.

The rows live in per-prefix YAML files (`llm_*.yaml`, `mgmt.yaml`, `mcp.yaml`,
`reliability.yaml`, `logging.yaml`, `guardrail.yaml`, `other.yaml`) and validate against
the discriminated union in `schema.py`, so an LLM row cannot carry a guardrail field and
vice versa. `logging` and `guardrail` are two id-prefixes that roll up into the single
"Logging & Guardrails" dashboard module.

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

The headline is P0 coverage. The collector also lists markers that point at ids not in
the registry, so a typo or an unenumerated behavior surfaces instead of being silently
dropped.

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
