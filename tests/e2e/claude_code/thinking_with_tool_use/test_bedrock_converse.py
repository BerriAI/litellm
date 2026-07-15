"""thinking_with_tool_use x Bedrock (Converse).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to AWS Bedrock via the `Converse` API, enable extended
thinking via `--effort high`, allow the built-in `Bash` tool, and
ask Claude to plan-and-execute a task that requires both reasoning and
a tool call. Assert the upstream returned both a `thinking` content
block and a `tool_use` content block in the same turn.

The Converse API has its own `additionalModelRequestFields.thinking`
shape and its own tool-use envelope; this cell catches gateway
regressions where the proxy fails to translate between Anthropic's
`thinking` parameter and Converse's reasoning configuration when tools
are also present.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/thinking_with_tool_use/test_bedrock_converse.py
                       ^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^
                       feature_id                  provider
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from claude_code._env import require_proxy
from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)


BEDROCK_CONVERSE_MODELS = [
    "claude-haiku-4-5-bedrock-converse",
    "claude-sonnet-4-6-bedrock-converse",
    "claude-opus-4-7-bedrock-converse",
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


def test_thinking_with_tool_use_bedrock_converse(compat_result):
    base_url, api_key = require_proxy(compat_result)

    outcomes = run_claude_models_parallel(
        models=BEDROCK_CONVERSE_MODELS,
        prompt=THINKING_TOOL_PROMPT,
        base_url=base_url,
        api_key=api_key,
        # thinking + tools combined into a single extra_args; see THINKING_ARGS
        extra_args=THINKING_ARGS + TOOL_USE_ARGS,
    )

    failures = []
    for model in BEDROCK_CONVERSE_MODELS:
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
