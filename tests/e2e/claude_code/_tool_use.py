"""Shared body for the `tool_use` / `tool_use_streaming` x <provider> compat cells.

Every tool_use cell follows the same skeleton:

  1. Read the proxy base URL + API key from env, fail-early if missing.
  2. Ask each model tier (fanned out via `run_claude_models_parallel`)
     to invoke the built-in `Bash` tool.
  3. Inspect each model's outcome and report one `compat_result` row per
     model; `ClaudeCLIError`, non-zero exit, and a missing `tool_use`
     content block are all per-model fails, everything else is a
     per-model pass.
  4. Surface a joined failure message via `pytest.fail(...)` so the
     pytest run also goes red.

The streaming variant (`verify_streaming=True`) adds the
`--include-partial-messages` CLI flag and additionally requires that
`input_json_delta` stream events were observed; zero deltas means the
proxy collapsed the streamed tool input into a single complete block or
stripped fine-grained tool streaming.

Security rationale for `TOOL_USE_ARGS`: the prompt asks for the exact
command `echo pong`, the `--allowed-tools` rule restricts the Bash tool
to that command, and `--permission-mode dontAsk` auto-denies anything
else the model returns instead of executing it. `dontAsk` mode in
headless `--print` mode only runs tools matching an explicit `allow`
rule (plus the built-in read-only set), so a compromised provider
response cannot turn the `Bash` allowlist into arbitrary host execution
(which would expose `docker inspect compat-proxy` /
`/proc/<proxy_pid>/environ` and thereby provider credentials living in
the proxy container).

The conftest infers `(feature_id, provider)` purely from the test file
path, so each per-provider file just declares its model list and calls
`run_tool_use_cell(...)`. This keeps all cell logic in one place; a
future tweak to the outcome checks or the failure-loop shape now
propagates to every cell automatically.

The leading underscore in the filename is what keeps pytest from
collecting this module as a test file.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import pytest

from claude_code._env import require_proxy
from claude_code.cli_driver import (
    ClaudeCLIError,
    DriverResult,
    failure_diagnostic,
    run_claude_models_parallel,
)


ClaudeRunner = Callable[..., Mapping[str, DriverResult | ClaudeCLIError]]

TOOL_USE_PROMPT = "Use the Bash tool to run the command `echo pong` and report what it printed."

TOOL_USE_ARGS = (
    "--allowed-tools",
    "Bash(echo pong)",
    "--permission-mode",
    "dontAsk",
)

TOOL_USE_STREAMING_ARGS = TOOL_USE_ARGS + ("--include-partial-messages",)


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
    inner_events = (event.get("event") for event in events if event.get("type") == "stream_event")
    return sum(
        1
        for inner in inner_events
        if isinstance(inner, Mapping)
        and inner.get("type") == "content_block_delta"
        and isinstance(inner.get("delta"), Mapping)
        and inner["delta"].get("type") == "input_json_delta"
    )


def run_tool_use_cell(
    *,
    compat_result,
    models: Sequence[str],
    verify_streaming: bool = False,
    env: Mapping[str, str] | None = None,
    runner: ClaudeRunner = run_claude_models_parallel,
) -> None:
    """Run the shared `tool_use` / `tool_use_streaming` x <provider> cell body.

    With ``verify_streaming=False`` the cell asserts that a `tool_use`
    content block came back over the wire. With ``verify_streaming=True``
    it additionally passes ``--include-partial-messages`` and asserts
    that `input_json_delta` stream events were observed for the block,
    which catches both known gateway regressions: buffering the streamed
    tool input into one complete block and stripping the
    fine-grained-tool-streaming beta so the upstream falls back to
    non-streaming tool_use.
    """
    base_url, api_key = require_proxy(compat_result, env=env)

    extra_args = TOOL_USE_STREAMING_ARGS if verify_streaming else TOOL_USE_ARGS

    outcomes = runner(
        models=models,
        prompt=TOOL_USE_PROMPT,
        base_url=base_url,
        api_key=api_key,
        extra_args=extra_args,
    )

    failures = []
    for model in models:
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
            error = f"[{model}] no tool_use content block observed in stream-json events"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if verify_streaming and _count_input_json_deltas(outcome.events) == 0:
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
