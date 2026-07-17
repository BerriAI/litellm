# e2e coverage registry

This directory holds the **denominator** for e2e test coverage: the set of behaviors we
want covered. The denominator is generated from the product surface rather than
hand-listed, so it grows on its own when the product gains a surface; a test PR only adds
a `@pytest.mark.covers(...)` marker. The naming grammar lives in `tests/e2e/CLAUDE.md`.

## Why it is generated

The denominator used to be a hand-written set of YAML rows, one per behavior. That is
self-fulfilling: the row and its covering test land in the same PR, so a wanted-but-untested
behavior never shows up as a gap. Deriving the denominator from the tests instead is just
as useless; it is 100% by construction. So the set of cells is derived from what the proxy
actually supports, and the tests are diffed against it.

## The model

A **cell** is one customer-noticeable behavior a single e2e test can assert pass/fail on,
for example `llm.chat_completions.bedrock_converse.tool_use.stream.works`. Cells are grouped
`module > feature > test`, with LLM cells split into `Core LLMs` and `Non-Core LLMs` for
dashboarding. The module, endpoint, route, capability and streaming are parsed back out of
the id (see `parse_llm_id` / `parse_module` in `schema.py`); nothing restates them, so an id
and its facets can never drift.

The denominator is the union of two sources.

Generated surface (`product_surface.py`). The conversational core (chat_completions,
messages, responses) is generated from the typed vocabularies in `schema.py` crossed with
the capability metadata in `model_prices_and_context_window.json`, the same file the proxy
ships. A flagged capability (tool_use, vision, thinking, structured_output, prompt caching)
is emitted for a route only when a model on that route advertises the matching `supports_*`
flag, so adding provider support in that json grows the denominator with no edit here. The
Anthropic-format `messages` surface is the Claude Code compatibility matrix, so its
capabilities are the CLI feature set and are not gated by model flags. The live route table
(`litellm.proxy._types.LiteLLMRoutes`) is read as a drift check: the collector warns when
the vocabulary enumerates an LLM endpoint the proxy no longer serves. This exact set of
product-surface sources is a design decision open to review.

Curated overlay (`overlay.yaml`). The only per-cell data a human decides lives here, keyed
by cell id: `tier`, `source`, `rationale`, `fail_before_fix`, and `supported`. The overlay
never decides whether a behavior exists; it annotates a generated cell, and it also
enumerates the ids that generation does not yet produce (the non-core LLM operations such as
batches, files, rerank, embeddings, audio and images, and the behavior modules mgmt, mcp,
reliability, quota, logging, guardrail, other, which have no clean cartesian to generate
from). A generated cell with no overlay row defaults to P2, so a newly generated surface
shows up as an uncovered gap rather than vanishing. Ordinary test PRs never touch this file;
it should be owner-gated via CODEOWNERS (not added here).

A test declares what it covers with a marker, and nothing else:

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

The headline is overall coverage. The collector also lists markers that point at ids not in
the denominator, so a typo or an unenumerated behavior surfaces instead of being silently
dropped, and warns on route-table drift.

Use strict mode in CI once existing markers are reconciled:

```
cd tests/e2e && PYTHONPATH=. python -m coverage_registry.collector --strict
```

Strict mode exits non-zero on `@pytest.mark.covers(...)` ids that are not in the denominator.
Add `--fail-on-collection-errors` when the job should also fail on pytest collection errors.

## Follow-up

Generation currently covers the conversational core only. The non-core LLM operations and
the behavior modules still live in the overlay because they have no clean product-surface
enumeration yet; wiring their own sources into generation (the route table for endpoint
operations, `guardrail_hooks/` for guardrail providers, `integrations/` for logging targets,
`router_strategy/` for reliability behaviors) is the remaining work, tracked rather than
faked. Tiers in the overlay were migrated from the previous hand-written rows and are a
first proposal, not a sign-off.
