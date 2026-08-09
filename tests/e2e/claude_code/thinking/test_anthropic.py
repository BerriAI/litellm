"""thinking x Anthropic.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
to Anthropic, enable extended thinking via `--effort high`, and assert
that the upstream returned a `thinking` content block. This proves the
proxy preserves Anthropic's `thinking` request parameter and the
upstream response's `thinking` content blocks end-to-end.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/thinking/test_anthropic.py
                       ^^^^^^^^      ^^^^^^^^^
                       feature_id    provider

The three Claude tiers run in parallel inside this single test, with
one `compat_result.add(...)` entry per model so the matrix builder
still sees three rows for this (feature, provider).
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

# --effort max maps to the largest thinking budget on every supported
# Claude tier; the test cares about wire shape, not answer quality. We
# use a CLI flag (rather than the legacy MAX_THINKING_TOKENS env var)
# because Claude Code 2.x reads thinking config from --effort, not from
# the env, and silently no-ops the env var. We use `max` rather than
# `high` because Sonnet 4.6 / Opus 4.7 only emit thinking blocks when
# the budget is generous and the prompt is non-trivial.
THINKING_ARGS = ["--effort", "max"]
# A puzzle non-trivial enough that Sonnet/Opus actually engage thinking
# rather than answer from memory. Trivial arithmetic ("3-2=?") is
# optimized away on the modern tiers and arrives without a thinking
# block, which would make this test silently false-fail under
# `--effort max`. Haiku 4.5 thinks even for trivial prompts; Sonnet 4.6
# and Opus 4.7 only emit thinking when the upstream judges it useful.
THINKING_PROMPT = (
    "I have a 3-gallon jug and a 5-gallon jug. How can I measure "
    "exactly 4 gallons of water? Think through the steps carefully."
)


def _has_thinking_block(events: Sequence[Mapping[str, Any]]) -> bool:
    """Walk the stream-json events and return True if any assistant
    message included a `thinking` content block."""
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                return True
    return False


@pytest.mark.covers("llm.messages.anthropic.thinking.nonstream.works")
def test_thinking_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy with thinking
    enabled and assert a `thinking` content block was emitted."""
    base_url, api_key = require_proxy(compat_result)

    outcomes = run_claude_models_parallel(
        models=ANTHROPIC_MODELS,
        prompt=THINKING_PROMPT,
        base_url=base_url,
        api_key=api_key,
        extra_args=THINKING_ARGS,
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

        if not _has_thinking_block(outcome.events):
            error = (
                f"[{model}] no `thinking` content block observed in stream-json events"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
