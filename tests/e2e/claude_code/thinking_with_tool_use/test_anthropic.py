"""thinking_with_tool_use x Anthropic.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
to Anthropic, enable extended thinking via `--effort high`, allow
the built-in `Bash` tool, and ask Claude to plan-and-execute a task
that requires both reasoning and a tool call. Assert the upstream
returned both a `thinking` content block and a `tool_use` content
block in the same turn — proving the proxy preserves the wire shape
where extended thinking and tool use coexist.

This is the cell that historically catches the most provider bugs:
"thinking blocks cannot be modified" 400s, the recurring Bedrock
"thinking.type.enabled is not supported" error, and the
`fine-grained-tool-streaming` + `interleaved-thinking` beta-header
interactions. A regression in any of those collapses this cell.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/thinking_with_tool_use/test_anthropic.py
                       ^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
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


ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-7",
]

# Extended thinking on, with a small budget — enough to surface a
# non-empty thinking block on a trivial reasoning prompt without
# blowing up wall time.
THINKING_ARGS = ["--effort", "max"]

# Prompt designed to force both blocks: the model has to *reason* about
# what command to run before *invoking* the Bash tool. Using a fixed
# expected output keeps the assertion focused on the wire shape rather
# than on answer quality.
# Prompt fixes the exact bash command to `echo pong`. The thinking
# block is preserved (the model reasons about why `echo pong` works),
# but the executed command is pinned so the cell can run under the
# tight `Bash(echo pong) + dontAsk` permission below — see
# `tool_use/test_anthropic.py` for the full security rationale.
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
    """Walk the stream-json events and return True if any assistant
    message included a content block of the given type."""
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


@pytest.mark.covers("llm.messages.anthropic.thinking_with_tool_use.nonstream.works")
def test_thinking_with_tool_use_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy with thinking
    enabled and tool use, and assert both `thinking` and `tool_use`
    content blocks landed in the same turn."""
    base_url, api_key = require_proxy(compat_result)

    outcomes = run_claude_models_parallel(
        models=ANTHROPIC_MODELS,
        prompt=THINKING_TOOL_PROMPT,
        base_url=base_url,
        api_key=api_key,
        # thinking + tools combined into a single extra_args; see THINKING_ARGS
        extra_args=THINKING_ARGS + TOOL_USE_ARGS,
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

        if not _has_block_type(outcome.events, "thinking"):
            error = (
                f"[{model}] no `thinking` content block observed; thinking "
                f"either disabled by the proxy or stripped by the upstream"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if not _has_block_type(outcome.events, "tool_use"):
            error = (
                f"[{model}] no `tool_use` content block observed alongside "
                f"thinking; the proxy may have dropped tools when thinking is on"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
