"""tool_search x Bedrock (Invoke).

HTTP-probe row. Sends a single `/v1/messages` request whose `tools`
array includes a `tool_search_tool_regex_20251119` discovery tool, and
asserts the proxy round-trips it to the upstream without a 400. This
verifies LiteLLM's tool-search beta-header translation
(`advanced-tool-use-2025-11-20` for Anthropic-shape providers,
`tool-search-tool-2025-10-19` for Vertex/Bedrock) survives end-to-end.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_search/test_bedrock_invoke.py
                       ^^^^^^^^^^^      ^^^^^^^^^^^^^^
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

import os

import pytest

from claude_code.http_probe import (
    assert_tool_search_shape,
    probe_tool_search,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

BEDROCK_INVOKE_MODELS = [
    "claude-haiku-4-5-bedrock-invoke",
    "claude-sonnet-5-bedrock-invoke",
    "claude-opus-4-8-bedrock-invoke",
]


def test_tool_search_bedrock_invoke(compat_result):
    """Probe `/v1/messages` with a `tool_search_tool_regex_20251119`
    tool and assert the proxy + upstream accept it for every Bedrock (Invoke)
    tier."""
    base_url = os.environ.get(PROXY_BASE_URL_ENV)
    api_key = os.environ.get(PROXY_API_KEY_ENV)
    if not base_url or not api_key:
        compat_result.set(
            {
                "status": "fail",
                "error": (
                    f"missing required env: set {PROXY_BASE_URL_ENV} and "
                    f"{PROXY_API_KEY_ENV} to point at a running LiteLLM proxy"
                ),
            }
        )
        pytest.fail(
            f"{PROXY_BASE_URL_ENV} / {PROXY_API_KEY_ENV} not configured",
            pytrace=False,
        )

    failures = []
    for model in BEDROCK_INVOKE_MODELS:
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
