"""extended_thinking x Azure (Microsoft Foundry).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to Anthropic's models hosted in Microsoft Foundry on
Azure, enable extended thinking via `MAX_THINKING_TOKENS`, and assert
that the upstream returned a `thinking` content block.

Foundry's Claude deployments advertise `supports_reasoning: true` in
LiteLLM's pricing metadata; the `thinking={"type": "enabled", ...}`
parameter passes through `azure_ai/claude-*` to Foundry's
`/anthropic/v1/messages` endpoint unchanged. Note that
`claude-opus-4-7-preview` documents thinking as not supported on
Foundry; if that lands, this row may flip to a partial pass and we'll
re-evaluate.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/extended_thinking/test_azure.py
                       ^^^^^^^^^^^^^^^^^      ^^^^^
                       feature_id             provider
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import pytest

from tests.claude_code.cli_driver import ClaudeCLIError, failure_diagnostic, run_claude

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-6-azure",
    "claude-opus-4-7-azure",
]

THINKING_ENV = {"MAX_THINKING_TOKENS": "4096"}
THINKING_PROMPT = (
    "Think step by step: if I have three apples and eat two, how many remain? "
    "Answer with the single digit only."
)


def _has_thinking_block(events: Sequence[Mapping[str, Any]]) -> bool:
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                return True
    return False


@pytest.mark.parametrize("model", AZURE_MODELS)
def test_extended_thinking_azure(compat_result, model):
    """Drive the `claude` CLI against the LiteLLM proxy with thinking
    enabled and assert a `thinking` content block was emitted."""
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
            prompt=THINKING_PROMPT,
            model=model,
            base_url=base_url,
            api_key=api_key,
            extra_env=THINKING_ENV,
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

    if not _has_thinking_block(result.events):
        compat_result.set(
            {
                "status": "fail",
                "error": f"[{model}] no `thinking` content block observed in stream-json events",
            }
        )
        pytest.fail(f"no thinking block for {model}", pytrace=False)
        return

    compat_result.set({"status": "pass"})
