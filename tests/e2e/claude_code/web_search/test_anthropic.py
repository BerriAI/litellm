"""web_search x Anthropic.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
to Anthropic, allow the built-in `WebSearch` tool, ask a question that
requires fresh web data, and assert that the upstream emitted a
`tool_use` block calling `WebSearch` — proving the proxy preserves
Claude Code's tool definitions and the upstream's tool-use response
end-to-end.

Note: Claude Code's `WebSearch` is a *client-side* tool (the CLI
executes the search itself and feeds the result back as a `tool_result`
block), so the wire shape is `tool_use` with `name="WebSearch"` rather
than the Anthropic-managed `server_tool_use` / `web_search_tool_result`
blocks (which only appear when the request includes the
`web_search_20250305` server tool definition — something the CLI does
not currently inject). A regression where the proxy strips the
`WebSearch` tool from the request or drops the `tool_use` block from
the response will break this assertion.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/web_search/test_anthropic.py
                       ^^^^^^^^^^      ^^^^^^^^^
                       feature_id      provider
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import pytest

from claude_code.cli_driver import (
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

# The CLI tool name surfaced as `tool_use.name` when WebSearch fires.
WEB_SEARCH_TOOL_NAME = "WebSearch"


def _has_web_search_tool_use(events: Sequence[Mapping[str, Any]]) -> bool:
    """Walk the stream-json events and return True if any assistant
    message included a `tool_use` block calling `WebSearch`."""
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
            if (
                block.get("type") == "tool_use"
                and block.get("name") == WEB_SEARCH_TOOL_NAME
            ):
                return True
    return False


@pytest.mark.covers("llm.messages.anthropic.web_search.nonstream.works")
def test_web_search_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert the
    upstream emitted a `tool_use` block calling `WebSearch`, proving
    the proxy preserved both the request-side tool definition and the
    response-side tool_use block."""
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

        if not _has_web_search_tool_use(outcome.events):
            error = (
                f"[{model}] no `tool_use` block with name=WebSearch observed; "
                "the proxy may have stripped the WebSearch tool definition from "
                "the request or the tool_use block from the response"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
