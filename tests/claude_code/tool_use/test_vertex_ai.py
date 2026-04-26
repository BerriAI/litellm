"""tool_use x Vertex AI.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to Anthropic's models on Google Cloud Vertex AI, ask
Claude to invoke a built-in tool (`Bash`), and assert that the upstream
returned a `tool_use` content block.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/tool_use/test_vertex_ai.py
                       ^^^^^^^^      ^^^^^^^^^
                       feature_id    provider
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import pytest

from tests.claude_code.cli_driver import ClaudeCLIError, failure_diagnostic, run_claude

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-6-vertex",
    "claude-opus-4-7-vertex",
]

TOOL_USE_PROMPT = (
    "Use the Bash tool to run the command `echo pong` and report what it printed."
)
TOOL_USE_ARGS = ["--allowed-tools", "Bash"]


def _has_tool_use_event(events: Sequence[Mapping[str, Any]]) -> bool:
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return True
    return False


@pytest.mark.parametrize("model", VERTEX_AI_MODELS)
def test_tool_use_vertex_ai(compat_result, model):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire."""
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

    try:
        result = run_claude(
            prompt=TOOL_USE_PROMPT,
            model=model,
            base_url=base_url,
            api_key=api_key,
            extra_args=TOOL_USE_ARGS,
        )
    except ClaudeCLIError as exc:
        compat_result.set({"status": "fail", "error": f"[{model}] {exc}"})
        pytest.fail(str(exc), pytrace=False)
        return

    if result.exit_code != 0:
        compat_result.set(
            {
                "status": "fail",
                "error": f"[{model}] claude CLI failed: {failure_diagnostic(result)}",
            }
        )
        pytest.fail(
            f"[{model}] claude CLI failed: {failure_diagnostic(result)}", pytrace=False
        )
        return

    if not _has_tool_use_event(result.events):
        compat_result.set(
            {
                "status": "fail",
                "error": f"[{model}] no tool_use content block observed in stream-json events",
            }
        )
        pytest.fail(f"no tool_use for {model}", pytrace=False)
        return

    compat_result.set({"status": "pass"})
