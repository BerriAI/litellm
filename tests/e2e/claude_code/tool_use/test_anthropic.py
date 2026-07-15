"""tool_use x Anthropic.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
to Anthropic, ask Claude to invoke a built-in tool (`Bash`), and assert
that the upstream returned a `tool_use` content block. This proves the
proxy preserves Claude Code's tool-call wire shape end-to-end.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_anthropic.py
                       ^^^^^^^^      ^^^^^^^^^
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

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

# Built-in tool-use prompt: ask Claude to use the `Bash` tool. The CLI
# allow-lists the tool via `--allowed-tools` so the run completes without
# an interactive permission prompt.
TOOL_USE_PROMPT = (
    "Use the Bash tool to run the command `echo pong` and report what it printed."
)
# Restrict the Bash tool to the exact command `echo pong` and put the
# CLI in `dontAsk` mode so anything else the model returns is auto-
# denied instead of executed. `dontAsk` mode in headless `--print` mode
# only runs tools matching an explicit `allow` rule (plus the built-in
# read-only set), so a compromised provider response cannot turn the
# `Bash` allowlist into arbitrary host execution (which would expose
# `docker inspect compat-proxy` / `/proc/<proxy_pid>/environ` and
# thereby provider credentials living in the proxy container).
TOOL_USE_ARGS = [
    "--allowed-tools",
    "Bash(echo pong)",
    "--permission-mode",
    "dontAsk",
]


def _has_tool_use_event(events: Sequence[Mapping[str, Any]]) -> bool:
    """Walk the stream-json events and return True if any assistant
    message included a `tool_use` content block."""
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


@pytest.mark.covers("llm.messages.anthropic.tool_use.nonstream.works")
def test_tool_use_anthropic(compat_result):
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

    outcomes = run_claude_models_parallel(
        models=ANTHROPIC_MODELS,
        prompt=TOOL_USE_PROMPT,
        base_url=base_url,
        api_key=api_key,
        extra_args=TOOL_USE_ARGS,
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
