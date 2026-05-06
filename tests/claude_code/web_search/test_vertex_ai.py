"""web_search x Vertex AI.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to GCP Vertex AI, allow the built-in `WebSearch` tool,
ask a question that requires fresh web data, and assert that the
upstream emitted a `server_tool_use` or `web_search_tool_result`
content block.

Per the changelog (1.0.110, 2.1.79+), Vertex has had inconsistent
support for the Anthropic web_search server tool — the cell will fail
loudly when the upstream rejects the tool, which is the diagnostic we
want for a compat matrix.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/web_search/test_vertex_ai.py
                       ^^^^^^^^^^      ^^^^^^^^^
                       feature_id      provider
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import pytest

from tests.claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-6-vertex",
    "claude-opus-4-7-vertex",
]

WEB_SEARCH_PROMPT = (
    "Use web search to find a news headline published this week about "
    "Anthropic. Reply with one sentence summarizing what you found."
)
WEB_SEARCH_ARGS = ["--allowed-tools", "WebSearch"]
SERVER_TOOL_BLOCK_TYPES = {"server_tool_use", "web_search_tool_result"}


def _has_server_tool_block(events: Sequence[Mapping[str, Any]]) -> bool:
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in SERVER_TOOL_BLOCK_TYPES:
                return True
    return False


def test_web_search_vertex_ai(compat_result):
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
            f"{PROXY_BASE_URL_ENV} / {PROXY_API_KEY_ENV} not configured", pytrace=False
        )

    outcomes = run_claude_models_parallel(
        models=VERTEX_AI_MODELS,
        prompt=WEB_SEARCH_PROMPT,
        base_url=base_url,
        api_key=api_key,
        extra_args=WEB_SEARCH_ARGS,
    )

    failures = []
    for model in VERTEX_AI_MODELS:
        outcome = outcomes[model]
        if isinstance(outcome, ClaudeCLIError):
            error = f"[{model}] {outcome}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if outcome.exit_code != 0:
            error = f"[{model}] claude CLI failed: {failure_diagnostic(outcome)}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if not _has_server_tool_block(outcome.events):
            error = (
                f"[{model}] no server_tool_use / web_search_tool_result block "
                "observed"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
