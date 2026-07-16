"""basic_messaging_streaming x Anthropic.

Drive the real `claude` CLI in headless `--output-format stream-json
--include-partial-messages` mode against a running LiteLLM proxy that
routes to Anthropic, and report the outcome via `compat_result`.

The cell goes green only when every Claude tier (a) returns a non-empty
reply and (b) the proxy actually streamed it — i.e. the CLI observed
multiple `stream_event` records carrying raw SSE deltas. A proxy that
buffers the upstream stream and returns a single non-streaming chunk
emits zero such records, which is the regression this row catches.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_streaming/test_anthropic.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id                     provider

The shared `run_basic_messaging_cell` helper fans the three Claude tiers
out in parallel inside this single test, with one
`compat_result.add(...)` entry per model so the matrix builder still
sees three rows for this (feature, provider).
"""

from __future__ import annotations

import pytest
from claude_code._basic_messaging import run_basic_messaging_cell

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-7",
]


@pytest.mark.covers("llm.messages.anthropic.basic.stream.works")
def test_basic_messaging_streaming_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty streamed reply (one row per Claude tier).
    """
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=ANTHROPIC_MODELS,
        prompt="Count from 1 to 5, one number per line.",
        verify_streaming=True,
    )
