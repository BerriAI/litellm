"""Shared body for the `basic_messaging_*` × <provider> compat cells.

Every basic_messaging cell follows the same skeleton:

  1. Read the proxy base URL + API key from env, fail-early if missing.
  2. Fan the three Claude tiers out via `run_claude_models_parallel`.
  3. Inspect each model's outcome and report one `compat_result` row per
     model — `ClaudeCLIError`, non-zero exit, and empty assistant text
     are all per-model fails; everything else is a per-model pass.
  4. Surface a joined failure message via `pytest.fail(...)` so the
     pytest run also goes red.

The streaming variant additionally passes `verify_streaming=True`,
which adds the `--include-partial-messages` CLI flag and asserts that
the proxy actually streamed the response (see the helper docstring for
the wire-level rationale).

The conftest infers `(feature_id, provider)` purely from the test file
path, so each per-provider file just declares its model list and calls
`run_basic_messaging_cell(...)`. This keeps all cell logic in one place
— a future tweak to the env-missing guard or the failure-loop shape
now propagates to every cell automatically.

The leading underscore in the filename is what keeps pytest from
collecting this module as a test file.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import pytest

from claude_code._env import CliKeyProvider, require_compat_cli_credentials
from claude_code.cli_driver import (
    ClaudeCLIError,
    DriverResult,
    failure_diagnostic,
    run_claude_models_parallel,
)


def _default_cli_key_provider() -> str | None:
    """Bind at call time so the fixture had a chance to populate its
    module-level slot before we read it; a top-level ``from ... import``
    would snapshot ``None`` at import time."""
    from claude_code.conftest import _compat_cli_key_provider

    return _compat_cli_key_provider()


ClaudeRunner = Callable[..., Mapping[str, DriverResult | ClaudeCLIError]]

# Floor on the number of `stream_event` records (with delta payloads)
# we expect to see when the proxy actually streams. With
# `--include-partial-messages`, the CLI emits one `stream_event` per
# raw upstream SSE event — a fully-streamed response produces many
# (`message_start`, multiple `content_block_delta`s, `content_block_stop`,
# `message_delta`, `message_stop`); a proxy that buffers the upstream
# and returns a single non-streaming chunk produces 0 or 1. Floor of 2
# is safely above the buffered case for any non-trivial reply, which
# is why the streaming cells use a "count from 1 to 5" style prompt.
MIN_STREAM_DELTA_EVENTS = 2


def _count_stream_event_deltas(events: Sequence[Mapping[str, Any]]) -> int:
    """Count `stream_event` records that carry an SSE event payload.

    With `--include-partial-messages`, Claude Code wraps every upstream
    SSE event in a `{"type": "stream_event", "event": {...}}` record.
    A buffering proxy collapses the upstream stream into a single
    non-streaming response, so these records vanish. Counting them
    (rather than just `len(events)`) is the wire-level signal that
    "did the proxy preserve streaming?" — independent of the `system`
    /`assistant`/`result` boilerplate records the CLI always emits.
    """
    count = 0
    for event in events:
        if event.get("type") != "stream_event":
            continue
        if isinstance(event.get("event"), Mapping):
            count += 1
    return count


def run_basic_messaging_cell(
    *,
    compat_result,
    models: Sequence[str],
    prompt: str,
    verify_streaming: bool = False,
    env: Mapping[str, str] | None = None,
    runner: ClaudeRunner = run_claude_models_parallel,
    cli_key_provider: CliKeyProvider = _default_cli_key_provider,
) -> None:
    """Run the shared `basic_messaging_*` × <provider> cell body.

    When ``verify_streaming=True``, the cell additionally asserts that
    the proxy streamed the response end-to-end. The check works by
    passing ``--include-partial-messages`` to the `claude` CLI, which
    causes it to emit one ``stream_event`` record per raw upstream SSE
    event (``message_start``, ``content_block_delta``, ``message_stop``,
    etc.). A proxy that buffers the upstream stream and returns a
    single non-streaming response collapses those records to zero —
    so a floor of ``MIN_STREAM_DELTA_EVENTS`` ``stream_event`` records
    catches the buffering regression without needing a streaming-aware
    driver.

    This is the same shape of check as ``tool_use_streaming`` uses,
    just keyed off the explicit partial-message flag so it works for
    plain assistant replies (where the CLI would otherwise collapse a
    streamed reply to a single ``assistant`` event in
    ``--print --output-format stream-json`` mode).
    """
    base_url, api_key = require_compat_cli_credentials(
        compat_result, cli_key_provider=cli_key_provider, env=env
    )

    extra_args: Sequence[str] = (
        ("--include-partial-messages",) if verify_streaming else ()
    )

    outcomes = runner(
        models=models,
        prompt=prompt,
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

        if not outcome.text.strip():
            error = f"[{model}] claude returned empty assistant text"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if verify_streaming:
            stream_event_count = _count_stream_event_deltas(outcome.events)
            if stream_event_count < MIN_STREAM_DELTA_EVENTS:
                error = (
                    f"[{model}] only {stream_event_count} stream_event records "
                    f"observed (< {MIN_STREAM_DELTA_EVENTS}); proxy likely "
                    f"buffered the upstream response instead of streaming it"
                )
                compat_result.add({"status": "fail", "error": error})
                failures.append(error)
                continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
