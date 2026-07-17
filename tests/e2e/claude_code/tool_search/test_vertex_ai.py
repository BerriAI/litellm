"""tool_search x Vertex AI.

HTTP-probe row. Sends a single `/v1/messages` request whose `tools`
array includes a `tool_search_tool_regex_20251119` discovery tool, and
asserts the proxy round-trips it to the upstream without a 400. This
verifies LiteLLM's tool-search beta-header translation
(`advanced-tool-use-2025-11-20` for Anthropic-shape providers,
`tool-search-tool-2025-10-19` for Vertex/Bedrock) survives end-to-end.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_search/test_vertex_ai.py
                       ^^^^^^^^^^^      ^^^^^^^^^
                       feature_id       provider

Why HTTP probe instead of CLI / MCP fan-out:

Real Claude Code activates tool_search by registering >N MCP tools and
relying on the model's internal heuristic to call the discovery tool
before any user tool. That setup requires standing up a stub MCP
server that exposes 50+ tool stubs and depends on Claude Code's
auto-deferral heuristic continuing to fire at today's tool count --
both of which break silently when Claude Code's threshold changes
between releases.

The bugs LiteLLM has actually shipped fixes for in this area
(2.1.117, 2.1.72, 2.1.70 in the Claude Code release notes) are
beta-header translation and proxy-side type recognition, not MCP
fan-out behavior. An HTTP probe hits exactly that surface: the
request goes out with a `tool_search_tool_regex_20251119` tool type,
the proxy is responsible for attaching the per-provider beta header
and forwarding, and the upstream either accepts or 400s. A red cell
here is always a proxy-side regression, not a flaky model-behavior
artifact.

Three Claude tiers are probed in sequence (count is too low to be
worth the parallelism overhead, and HTTP probes don't compete for
the proxy's `--num-workers` slots the way CLI subprocess runs do).
The matrix's "all three must pass" rule still applies via the
per-cell aggregator.
"""

from __future__ import annotations

import pytest

from claude_code._env import require_proxy
from claude_code.http_probe import (
    assert_tool_search_shape,
    probe_tool_search,
)


VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-5-vertex",
    "claude-opus-4-7-vertex",
]


@pytest.mark.skip(reason="stage red: Vertex rejects tool_search when deployment extra_headers inject context-1m beta; product/config")
@pytest.mark.covers("llm.messages.vertex.tool_search.nonstream.works")
def test_tool_search_vertex_ai(compat_result):
    """Probe `/v1/messages` with a `tool_search_tool_regex_20251119`
    tool and assert the proxy + upstream accept it for every Vertex AI
    tier."""
    base_url, api_key = require_proxy(compat_result)

    failures = []
    for model in VERTEX_AI_MODELS:
        result = probe_tool_search(base_url=base_url, api_key=api_key, model=model)
        shape_error = assert_tool_search_shape(result)
        if shape_error is not None:
            error = f"[{model}] tool_search probe failed: {shape_error}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
