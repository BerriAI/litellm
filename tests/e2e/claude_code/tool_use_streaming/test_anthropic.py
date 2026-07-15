"""tool_use_streaming x Anthropic.

Drive the real `claude` CLI in headless `--output-format stream-json`
mode (with `--include-partial-messages`) against a running LiteLLM
proxy that routes to Anthropic, ask Claude to invoke a built-in tool
(`Bash`), and assert that the upstream (a) emitted a `tool_use` content
block and (b) actually streamed the tool input incrementally — i.e.
`input_json_delta` stream events were observed for the block.

This is the "fine-grained tool streaming" path. Historically gateways
break it in two ways: they either buffer/collapse the streamed tool
input into a single complete block (no `input_json_delta` records
reach the client) or they strip the
`fine-grained-tool-streaming-2025-05-14` beta header and the upstream
falls back to non-streaming tool_use. Both regressions are caught by
the assertions below.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_anthropic.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id              provider
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
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

# Same shape as the non-streaming `tool_use` cell: ask Claude to call
# the built-in `Bash` tool. `--include-partial-messages` surfaces the
# raw SSE records as `stream_event` entries in the stream-json output,
# which is the wire-level signal for whether the proxy preserved
# incremental `input_json_delta` events for the tool_use block.
TOOL_USE_PROMPT = (
    "Use the Bash tool to run the command `echo pong` and report what it printed."
)
# Bash is restricted to the exact command `echo pong` + `dontAsk`
# permission mode; see `tool_use/test_anthropic.py` for the security
# rationale.
TOOL_USE_ARGS = [
    "--allowed-tools",
    "Bash(echo pong)",
    "--permission-mode",
    "dontAsk",
    "--include-partial-messages",
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


def _count_input_json_deltas(events: Sequence[Mapping[str, Any]]) -> int:
    """Count `input_json_delta` records among the `stream_event`
    entries. Zero means the proxy collapsed the streamed tool input
    into a single complete block instead of forwarding the incremental
    deltas the upstream emitted."""
    inner_events = (
        event.get("event") for event in events if event.get("type") == "stream_event"
    )
    return sum(
        1
        for inner in inner_events
        if isinstance(inner, Mapping)
        and inner.get("type") == "content_block_delta"
        and isinstance(inner.get("delta"), Mapping)
        and inner["delta"].get("type") == "input_json_delta"
    )


def test_tool_use_streaming_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert the
    proxy preserves fine-grained tool streaming end-to-end."""
    base_url, api_key = require_proxy(compat_result)

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

        if _count_input_json_deltas(outcome.events) == 0:
            error = (
                f"[{model}] no input_json_delta stream events observed; proxy "
                f"likely buffered the tool input into a complete block or "
                f"stripped fine-grained tool streaming"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
