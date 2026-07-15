"""tool_use_streaming x Vertex AI.

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Claude requests to
GCP Vertex AI, ask Claude to invoke a built-in tool (`Bash`), and
assert that the upstream (a) emitted a `tool_use` content block and
(b) actually streamed events incrementally.

Vertex AI exposes Anthropic models via `:streamRawPredict`; this cell
catches gateway regressions where the proxy buffers the response or
strips the streaming beta header on the way to Vertex.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_vertex_ai.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id              provider
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

VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-6-vertex",
    "claude-opus-4-7-vertex",
]

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


def test_tool_use_streaming_vertex_ai(compat_result):
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
        models=VERTEX_AI_MODELS,
        prompt=TOOL_USE_PROMPT,
        base_url=base_url,
        api_key=api_key,
        extra_args=TOOL_USE_ARGS,
    )

    failures = []
    for model in VERTEX_AI_MODELS:
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
