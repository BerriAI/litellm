"""tool_use x OpenAI (GPT-5.6).

Drive the real `claude` CLI against a running LiteLLM proxy that
routes Anthropic Messages requests to OpenAI's GPT-5.6 family (Sol,
Terra, Luna), ask the model to invoke a built-in tool (`Bash`), and
assert that a `tool_use` content block came back over the wire.

Claude Code declares its tools in Anthropic `tools` format; LiteLLM's
`openai/gpt-*` route translates them to OpenAI function calling and
maps the returned `tool_calls` back to Anthropic `tool_use` blocks, so
this cell exercises the tool-schema translation in both directions.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `tool_use/test_anthropic.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_openai.py
                       ^^^^^^^^      ^^^^^^
                       feature_id    provider
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

OPENAI_MODELS = [
    "gpt-5-6-sol-openai",
    "gpt-5-6-terra-openai",
    "gpt-5-6-luna-openai",
]

TOOL_USE_PROMPT = (
    "Use the Bash tool to run the command `echo pong` and report what it printed."
)
TOOL_USE_ARGS = [
    "--allowed-tools",
    "Bash(echo pong)",
    "--permission-mode",
    "dontAsk",
]


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


def test_tool_use_openai(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire by each GPT-5.6 tier."""
    proxy = require_proxy(compat_result)

    outcomes = run_claude_models_parallel(
        models=OPENAI_MODELS,
        prompt=TOOL_USE_PROMPT,
        base_url=proxy.base_url,
        api_key=proxy.api_key,
        extra_args=TOOL_USE_ARGS,
    )

    failures = []
    for model in OPENAI_MODELS:
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

        if not _has_tool_use_event(outcome.events):
            error = (
                f"[{model}] no tool_use content block observed in stream-json events"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
