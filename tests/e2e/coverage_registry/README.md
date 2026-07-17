# e2e coverage registry

This directory produces the **denominator** for e2e test coverage: the set of behaviors
we want covered. The denominator is derived from the product surface, not hand-listed, so
it grows itself when the product gains a surface and cannot be gamed by adding a row in the
same PR that adds the test. The naming grammar lives in `tests/e2e/CLAUDE.md`.

## The model

A **cell** is one customer-noticeable behavior a single e2e test can assert pass/fail on,
for example `llm.chat_completions.bedrock_converse.tool_use.stream.works`. A cell is
identified by a dotted id whose first segment is the module. The structural facets an LLM
id encodes (endpoint, route, capability, streaming) are parsed back out of the id rather
than stored a second time, so an id and its fields can never drift. Cells roll up
`module > feature > test`, with LLM cells split into `Core LLMs` and `Non-Core LLMs` for
dashboarding, and `logging` + `guardrail` folded into a single `Logging & Guardrails`
module.

The denominator is built in `registry.py` as two parts unioned:

```
denominator = generate_llm_cell_ids()     # product surface, self-growing
              | frozenset(load_overlay())  # ids generation does not yet produce
```

`product_surface.py` generates the conversational core (chat_completions, messages,
responses) by crossing the typed vocabularies in `schema.py` with the capability metadata
in `model_prices_and_context_window.json`, the same file the proxy ships. A flagged
capability (tool_use, vision, thinking, ...) is emitted for a route only when a model on
that route advertises the matching `supports_*` flag, so support added in that json grows
the denominator with no edit here. The Anthropic-format `messages` surface is the Claude
Code compatibility matrix, so its capabilities are the CLI feature set rather than model
flags and are intentionally ungated.

`overlay.yaml` is the small curated human overlay, keyed by cell id. It never decides
whether a behavior exists; it only annotates a generated cell with judgement (tier,
source, rationale, fail-before-fix, support) and enumerates the ids generation does not yet
produce (the non-core LLM operations and the behavior modules). A generated cell with no
overlay row defaults to P2, so a newly grown surface shows up as an uncovered gap rather
than vanishing. Ordinary test PRs never touch `overlay.yaml`; adding coverage is only a
marker:

```python
@pytest.mark.covers("llm.chat_completions.openai.tool_use.stream.works")
def test_openai_streaming_tool_calls(self) -> None:
    ...
```

## The number

`collector.py` diffs the denominator against those markers and reports coverage per module.
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

This emits exactly one `COVERAGE_TOTAL` line and one `COVERAGE_MODULE` line per module in
`MODULE_ORDER`, in that order. Loki uses log-safe `module=` labels from `LOKI_MODULE_LABELS`
(`core_llms`, `management_ui`, etc.) so existing JSON and Prometheus consumers keep their
human-readable module names unchanged.

The headline is overall coverage. The text format also lists markers that point at ids not
in the denominator (a typo or an unenumerated behavior), and warns when the schema
enumerates an LLM endpoint the live proxy route table no longer serves, so drift surfaces
instead of being silently dropped.

Use strict mode in CI once existing draft markers are reconciled:

```
cd tests/e2e && PYTHONPATH=. python -m coverage_registry.collector --strict
```

Strict mode exits non-zero on `@pytest.mark.covers(...)` ids that are not in the
denominator. Add `--fail-on-collection-errors` when the job should also fail on pytest
collection errors.

## Follow-up

Generation covers the conversational core only. The non-core LLM operations (batches,
files, rerank, embeddings, audio, images) use an operation grammar the vocabulary does not
enumerate, and the behavior modules (mgmt, mcp, reliability, quota, logging, guardrail,
other) have no clean cartesian; both are carried by the overlay for now. Wiring their own
product-surface sources (the route table for endpoint operations, `guardrail_hooks/`,
`integrations/`, `router_strategy/`) into generation is the remaining work. The live route
table (`litellm.proxy._types.LiteLLMRoutes`) is already read as a drift check rather than a
generation input, to keep that heavyweight import off the collector's hot path.
