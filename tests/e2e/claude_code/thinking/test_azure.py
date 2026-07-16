"""thinking x Azure (Microsoft Foundry).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to Anthropic's models hosted in Microsoft Foundry on
Azure, enable extended thinking via `--effort high`, and assert
that the upstream returned a `thinking` content block.

Foundry's Claude deployments advertise `supports_reasoning: true` in
LiteLLM's pricing metadata; the `thinking={"type": "enabled", ...}`
parameter passes through `azure_ai/claude-*` to Foundry's
`/anthropic/v1/messages` endpoint unchanged.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/thinking/test_azure.py
                       ^^^^^^^^      ^^^^^
                       feature_id    provider
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

AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-6-azure",
    "claude-opus-4-8-azure",
]

THINKING_ARGS = ["--effort", "max"]
THINKING_PROMPT = (
    "I have a 3-gallon jug and a 5-gallon jug. How can I measure "
    "exactly 4 gallons of water? Think through the steps carefully."
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


def test_thinking_azure(compat_result):
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

    outcomes = run_claude_models_parallel(
        models=AZURE_MODELS,
        prompt=THINKING_PROMPT,
        base_url=base_url,
        api_key=api_key,
        extra_args=THINKING_ARGS,
    )

    failures = []
    for model in AZURE_MODELS:
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

        if not _has_thinking_block(outcome.events):
            error = (
                f"[{model}] no `thinking` content block observed in stream-json events"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
