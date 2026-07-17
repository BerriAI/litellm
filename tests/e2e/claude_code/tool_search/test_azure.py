"""tool_search x Azure (Microsoft Foundry).

HTTP-probe row. Sends a single `/v1/messages` request whose `tools`
array includes a `tool_search_tool_regex_20251119` discovery tool, and
asserts the proxy round-trips it to the upstream without a 400. This
verifies LiteLLM's tool-search beta-header translation
(`advanced-tool-use-2025-11-20` for Anthropic-shape providers,
`tool-search-tool-2025-10-19` for Vertex/Bedrock) survives end-to-end.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_search/test_azure.py
                       ^^^^^^^^^^^      ^^^^^
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

from claude_code._env import require_gateway
from claude_code.http_probe import (
    assert_tool_search_shape,
    probe_tool_search,
)


AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-5-azure",
    "claude-opus-4-7-azure",
]


@pytest.mark.skip(reason="stage red: Azure Foundry tool_search_server not supported in workspace for probed models")
@pytest.mark.covers("llm.messages.azure_foundry.tool_search.nonstream.works")
def test_tool_search_azure(compat_result):
    """Probe `/v1/messages` with a `tool_search_tool_regex_20251119`
    tool and assert the proxy + upstream accept it for every Azure (Microsoft Foundry)
    tier."""
    gateway = require_gateway(compat_result)

    failures = []
    for model in AZURE_MODELS:
        result = probe_tool_search(gateway=gateway, model=model)
        shape_error = assert_tool_search_shape(result)
        if shape_error is not None:
            error = f"[{model}] tool_search probe failed: {shape_error}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
