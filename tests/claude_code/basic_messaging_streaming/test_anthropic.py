"""basic_messaging_streaming x Anthropic.

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes to Anthropic, and
report the outcome via `compat_result`.

The CLI is run with `--print --output-format stream-json`, which streams
incremental events as the upstream produces tokens. The cell goes green
only when every Claude tier returns a non-empty reply over a streamed
wire (i.e. at least one stream-json event is observed). This catches
regressions where the proxy buffers the full response before flushing,
silently degrading the streaming experience customers rely on.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/basic_messaging_streaming/test_anthropic.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id                     provider

The shared `run_basic_messaging_cell` helper fans the three Claude tiers
out in parallel inside this single test, with one
`compat_result.add(...)` entry per model so the matrix builder still
sees three rows for this (feature, provider). The `require_stream_events`
flag adds the streaming-only assertion that at least one stream-json
event was observed per model.
"""

from __future__ import annotations

from tests.claude_code._basic_messaging import run_basic_messaging_cell

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


def test_basic_messaging_streaming_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty streamed reply (at least one stream-json event observed).
    """
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=ANTHROPIC_MODELS,
        prompt="Count from 1 to 5, one number per line.",
        require_stream_events=True,
    )
