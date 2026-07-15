"""thinking_with_tool_use x Bedrock (Invoke).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to AWS Bedrock via the legacy `InvokeModel` API path,
enable extended thinking via `--effort high`, allow the built-in
`Bash` tool, and ask Claude to plan-and-execute a task that requires
both reasoning and a tool call. Assert the upstream returned both a
`thinking` content block and a `tool_use` content block in the same
turn.

This is the cell most likely to flush out Bedrock-specific bugs in
LiteLLM's Anthropic <-> Bedrock translation: the recurring
"thinking.type.enabled is not supported" 400 error has reappeared on
several Bedrock model routes (notably application inference profile
ARNs), and the only reliable signal that the fix is wired through the
proxy is a successful round-trip on this cell.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/thinking_with_tool_use/test_bedrock_invoke.py
                       ^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id                  provider
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

BEDROCK_INVOKE_MODELS = [
    "claude-haiku-4-5-bedrock-invoke",
    "claude-sonnet-4-6-bedrock-invoke",
    "claude-opus-4-7-bedrock-invoke",
]

THINKING_ARGS = ["--effort", "max"]
# Prompt + Bash restriction pin the executed command to `echo pong`;
# see `tool_use/test_anthropic.py` for the security rationale.
THINKING_TOOL_PROMPT = (
    "Think step by step about why the command `echo pong` prints just the "
    "word 'pong'. Then use the Bash tool to run exactly the command "
    "`echo pong` and report what it printed."
)
TOOL_USE_ARGS = [
    "--allowed-tools",
    "Bash(echo pong)",
    "--permission-mode",
    "dontAsk",
]


def _has_block_type(
    events: Sequence[Mapping[str, Any]],
    block_type: str,
) -> bool:
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == block_type:
                return True
    return False


@pytest.mark.covers("llm.messages.bedrock_invoke.thinking_with_tool_use.nonstream.works")
def test_thinking_with_tool_use_bedrock_invoke(compat_result):
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
        models=BEDROCK_INVOKE_MODELS,
        prompt=THINKING_TOOL_PROMPT,
        base_url=base_url,
        api_key=api_key,
        # thinking + tools combined into a single extra_args; see THINKING_ARGS
        extra_args=THINKING_ARGS + TOOL_USE_ARGS,
    )

    failures = []
    for model in BEDROCK_INVOKE_MODELS:
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

        if not _has_block_type(outcome.events, "thinking"):
            error = f"[{model}] no `thinking` content block observed"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if not _has_block_type(outcome.events, "tool_use"):
            error = f"[{model}] no `tool_use` content block observed alongside thinking"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
