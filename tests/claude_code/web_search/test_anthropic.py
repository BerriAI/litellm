"""web_search x Anthropic.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
to Anthropic, allow the built-in `WebSearch` tool, ask a question that
requires fresh web data, and assert that the upstream emitted a
`server_tool_use` or `web_search_tool_result` content block — proving
the proxy preserves Anthropic's server-side web search end-to-end.

Web search is a *server tool*: unlike `Bash`/`Read`/etc., the upstream
executes the search itself and embeds the results inline in the
response. The wire shape is distinctive:

  - `server_tool_use` block with `name: "web_search"`
  - `web_search_tool_result` block carrying the encrypted result content

A regression where the proxy strips the `web_search_20250305` tool
from the request, drops the result block from the response, or fails
to forward the required beta header collapses both signals.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/web_search/test_anthropic.py
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

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

# A prompt the model cannot answer from training data alone — it forces
# the model to actually hit the web_search server tool rather than
# replying from memory. We pick "this week" as the freshness anchor
# because it's stable across long-running test schedules without
# pinning to a specific date that would go stale.
WEB_SEARCH_PROMPT = (
    "Use web search to find a news headline published this week about "
    "Anthropic. Reply with one sentence summarizing what you found."
)
# Allow only WebSearch so the model has no fallback path: if the proxy
# strips the server tool, the run will fail loudly rather than silently
# answering from training data via a different tool.
WEB_SEARCH_ARGS = ["--allowed-tools", "WebSearch"]

# Block types that prove the server tool actually executed end-to-end.
SERVER_TOOL_BLOCK_TYPES = {"server_tool_use", "web_search_tool_result"}


def _has_server_tool_block(events: Sequence[Mapping[str, Any]]) -> bool:
    """Walk the stream-json events and return True if any assistant
    message included a server-tool block (server_tool_use or
    web_search_tool_result)."""
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


def test_web_search_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert the
    upstream emitted a server-tool block proving web search ran."""
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
        models=ANTHROPIC_MODELS,
        prompt=WEB_SEARCH_PROMPT,
        base_url=base_url,
        api_key=api_key,
        extra_args=WEB_SEARCH_ARGS,
    )

    failures = []
    for model in ANTHROPIC_MODELS:
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
                "observed; the proxy may have stripped the WebSearch server tool "
                "or its result block"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
