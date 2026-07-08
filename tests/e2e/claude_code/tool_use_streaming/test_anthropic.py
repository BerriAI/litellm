"""tool_use_streaming x Anthropic.

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes to Anthropic, ask
Claude to invoke a built-in tool (`Bash`), and assert that the upstream
(a) emitted a `tool_use` content block and (b) actually streamed the
events incrementally — i.e. more than one stream-json record was
observed before the final `result`.

This is the "fine-grained tool streaming" path. Historically gateways
break it in two ways: they either buffer the entire response before
flushing (in which case `len(events)` collapses to ~1 final record) or
they strip the `fine-grained-tool-streaming-2025-05-14` beta header and
the upstream falls back to non-streaming tool_use. Both regressions are
caught by the assertions below.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_anthropic.py
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

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

# Same shape as the non-streaming `tool_use` cell: ask Claude to call
# the built-in `Bash` tool. The CLI is already in stream-json mode by
# default in `run_claude`, so we don't need to toggle anything to
# exercise the streaming wire — what we want to assert is that the
# stream-json transport actually carried more than one record, which
# is the wire-level signal that the proxy didn't buffer the upstream.
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
]

# Floor on the number of stream-json records we expect to see for a
# tool-use turn. A buffered (non-streamed) wire for this multi-turn
# flow collapses to roughly: one `system` init + one `assistant` with
# the `tool_use` block + a `user` tool_result + one `assistant` final
# text + one `result`, i.e. ~5 records (the CLI executes the tool
# locally and sends the result back, producing a second model turn
# even on a fully buffered proxy). Real fine-grained streaming
# produces many more (incremental input_json_delta events,
# intermediate assistant deltas, etc., typically 15+). We pick a
# floor comfortably above the buffered case so the assertion catches
# the regression without being flaky on short responses.
MIN_STREAM_EVENTS = 8


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


def test_tool_use_streaming_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert the
    proxy preserves fine-grained tool streaming end-to-end."""
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

        if len(outcome.events) < MIN_STREAM_EVENTS:
            error = (
                f"[{model}] only {len(outcome.events)} stream-json events observed "
                f"(< {MIN_STREAM_EVENTS}); proxy likely buffered the response or "
                f"stripped fine-grained tool streaming"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
