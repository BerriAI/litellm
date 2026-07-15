"""tool_use x AWS Bedrock Mantle (GPT-5.6).

Drive the real `claude` CLI against a running LiteLLM proxy that
routes Anthropic Messages requests to OpenAI's GPT-5.6 family (Sol,
Terra, Luna) on AWS Bedrock's Mantle endpoint, ask the model to invoke
a built-in tool (`Bash`), and assert that a `tool_use` content block
came back over the wire.

Mantle speaks the OpenAI Responses API, whose tool declarations and
`function_call` outputs differ from both Anthropic Messages and
chat completions; LiteLLM's `bedrock_mantle/openai.gpt-*` route
translates Anthropic `tools` into Responses tool declarations and maps
the emitted function calls back to `tool_use` blocks.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `tool_use/test_anthropic.py` for the
security rationale.

GPT cells are opt-in via COMPAT_GPT_CELLS=1 (see
`claude_code._gpt_cells`).

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_bedrock_mantle.py
                       ^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id    provider
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import pytest

from claude_code._gpt_cells import skip_unless_gpt_cells_enabled
from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

BEDROCK_MANTLE_MODELS = [
    "gpt-5-6-sol-bedrock-mantle",
    "gpt-5-6-terra-bedrock-mantle",
    "gpt-5-6-luna-bedrock-mantle",
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


def test_tool_use_bedrock_mantle(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire by each GPT-5.6 tier."""
    skip_unless_gpt_cells_enabled()
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
        models=BEDROCK_MANTLE_MODELS,
        prompt=TOOL_USE_PROMPT,
        base_url=base_url,
        api_key=api_key,
        extra_args=TOOL_USE_ARGS,
    )

    failures = []
    for model in BEDROCK_MANTLE_MODELS:
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
